{{- define "nrc.name" -}}nrc-marketing{{- end -}}

{{- define "nrc.labels" -}}
app.kubernetes.io/part-of: nrc-marketing
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ required "image.tag is required (the git commit)" .Values.image.tag | quote }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "nrc.selector" -}}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/* Workload names: fixed in an environment that already has them (staging: marketing-api). */}}
{{- define "nrc.apiName" -}}{{ .Values.api.name | default (printf "%s-api" .Release.Name) }}{{- end -}}
{{- define "nrc.workerName" -}}{{ .Values.worker.name | default (printf "%s-worker" .Release.Name) }}{{- end -}}

{{- define "nrc.image" -}}
{{ required "image.repository is required" .Values.image.repository }}:{{ required "image.tag is required (the git commit)" .Values.image.tag }}
{{- end -}}

{{/* envFrom for one workload: the shared ConfigMap (unless skipped) plus its Secrets. */}}
{{- define "nrc.envFrom" -}}
{{- if not .skipConfig }}
- configMapRef:
    name: {{ .root.Release.Name }}-env
{{- end }}
{{- range .secrets }}
- secretRef:
    name: {{ . }}
{{- end }}
{{- end -}}
