"""Replay or settle a generation job that waits for an operator, with an audit row.

Find the jobs with `uv run python -m scripts.list_unknown_jobs` (owner, read-only), check each
one with the provider (the job id is in its idempotency key), then run one of:

    uv run python -m scripts.job_operations replay --tenant <uuid> --job <uuid> \
        --operator <id> --reason <text>
    uv run python -m scripts.job_operations settle --tenant <uuid> --job <uuid> \
        (--billed | --not-billed) [--actual-usd <decimal>] --operator <id> --reason <text>

replay: an UNKNOWN job the worker gave up on gets a new run event and a fresh deadline. The worker
reconciles with the job's own provider idempotency key, so the provider never bills it twice.
settle: record what the provider did. Billed commits the credits and spend, not billed releases
both. An UNKNOWN job becomes FAILED (OPERATOR_SETTLED_BILLED or OPERATOR_SETTLED_NOT_BILLED); a
CANCELLED job with HELD credits stays CANCELLED.

Writes connect as marketing_worker (WORKER_DATABASE_URL), the least privilege that can do both.
That role cannot read tenants, so --tenant is required. Every action writes one audit row
(job_operator_actions) in the same transaction. Exit code 0 on success, 2 when the operation is
refused or the arguments are invalid. Prints ids and codes only, never the reason.

--operator is self-declared: nothing checks it against a login. Whoever can use
WORKER_DATABASE_URL (a secret in AWS Secrets Manager; AWS logs reads of it) can act, so access to
that secret is the authorization. The audit row records the declared operator plus the change.
"""

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation
from typing import Final
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings, get_settings
from app.core.errors import AppError, JobOperationNotAllowedError, NotFoundError
from app.db.session import TenantSession
from app.models.jobs import OPERATOR_MAX_CHARS, REASON_MAX_CHARS
from app.services import job_operations
from app.services.spend import usd_micros
from app.workers.common import WorkerConfigError, worker_database_url

EXIT_OK: Final = 0
EXIT_REFUSED: Final = 2
# Sanity bound on --actual-usd (a typo such as 94000 for 0.094 would otherwise be recorded).
MAX_ACTUAL_USD: Final = Decimal(1000)


def _uuid(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a UUID") from None


def _text(name: str, max_chars: int) -> Callable[[str], str]:
    def parse(value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise argparse.ArgumentTypeError(f"{name} must not be empty")
        if len(stripped) > max_chars:
            raise argparse.ArgumentTypeError(f"{name} must be at most {max_chars} characters")
        return stripped

    return parse


def _usd(value: str) -> Decimal:
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise argparse.ArgumentTypeError("must be a decimal number") from None
    if not amount.is_finite() or amount < 0 or amount > MAX_ACTUAL_USD:
        raise argparse.ArgumentTypeError(f"must be between 0 and {MAX_ACTUAL_USD}")
    return amount


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="job_operations", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("replay", help="give an UNKNOWN job a new run event")
    settle = commands.add_parser("settle", help="record the provider's outcome and settle credits")
    for command in (replay, settle):
        command.add_argument("--tenant", type=_uuid, required=True)
        command.add_argument("--job", type=_uuid, required=True)
        command.add_argument(
            "--operator",
            type=_text("operator", OPERATOR_MAX_CHARS),
            required=True,
            help="your operator id; self-declared, not checked against a login",
        )
        command.add_argument("--reason", type=_text("reason", REASON_MAX_CHARS), required=True)
    outcome = settle.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--billed", action="store_true")
    outcome.add_argument("--not-billed", action="store_true")
    settle.add_argument("--actual-usd", type=_usd, default=None)
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    # Invalid arguments exit with code 2 (argparse), the same as a refused operation.
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "settle" and args.actual_usd is not None and args.not_billed:
        parser.error("--actual-usd only goes with --billed")
    return args


def _engine(database_url: str) -> AsyncEngine:
    # One short-lived connection per run, so no pool outlives the script.
    return create_async_engine(database_url, poolclass=NullPool)


async def replay_job(
    database_url: str,
    tenant_id: UUID,
    job_id: UUID,
    *,
    operator: str,
    reason: str,
    settings: Settings,
) -> UUID:
    engine = _engine(database_url)
    try:
        async with AsyncSession(engine, sync_session_class=TenantSession) as db:
            return await job_operations.replay(
                db, tenant_id, job_id, operator=operator, reason=reason, settings=settings
            )
    finally:
        await engine.dispose()


async def settle_job(
    database_url: str,
    tenant_id: UUID,
    job_id: UUID,
    *,
    billed: bool,
    actual_usd: Decimal | None,
    operator: str,
    reason: str,
) -> None:
    engine = _engine(database_url)
    try:
        async with AsyncSession(engine, sync_session_class=TenantSession) as db:
            await job_operations.settle(
                db,
                tenant_id,
                job_id,
                billed=billed,
                actual_usd_micros=None if actual_usd is None else usd_micros(actual_usd),
                operator=operator,
                reason=reason,
            )
    finally:
        await engine.dispose()


async def run(args: argparse.Namespace, database_url: str, settings: Settings) -> int:
    """Run the parsed command. Returns the exit code; prints ids and codes only."""
    try:
        if args.command == "replay":
            event_id = await replay_job(
                database_url,
                args.tenant,
                args.job,
                operator=args.operator,
                reason=args.reason,
                settings=settings,
            )
            print(f"replayed tenant={args.tenant} job={args.job} event={event_id}")
        else:
            await settle_job(
                database_url,
                args.tenant,
                args.job,
                billed=args.billed,
                actual_usd=args.actual_usd,
                operator=args.operator,
                reason=args.reason,
            )
            print(f"settled tenant={args.tenant} job={args.job} billed={args.billed}")
    except JobOperationNotAllowedError as error:
        details = error.details or {}
        print(
            f"refused code={error.code} tenant={args.tenant} job={args.job}"
            f" state={details.get('state')} reason={details.get('reason')}"
        )
        return EXIT_REFUSED
    except NotFoundError as error:
        print(f"refused code={error.code} tenant={args.tenant} job={args.job}")
        return EXIT_REFUSED
    except AppError as error:
        print(f"failed code={error.code} tenant={args.tenant} job={args.job}")
        return EXIT_REFUSED
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        url = worker_database_url().get_secret_value()
    except WorkerConfigError as error:
        print(error)
        return EXIT_REFUSED
    return asyncio.run(run(args, url, get_settings()))


if __name__ == "__main__":
    sys.exit(main())
