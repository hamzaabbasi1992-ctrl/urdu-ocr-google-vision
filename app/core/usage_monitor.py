"""Cross-cutting: real Google-confirmed Vision API usage, independent of
this app's own local page-count estimate.

The local estimate in app/simple_gui.py only counts what THIS app believes
it sent - wrong after a crash, or if the same credentials are ever used by
another tool. Cloud Monitoring's `serviceruntime.googleapis.com/api/request_count`
metric is Google's own authoritative record of API requests received, which
is what billing is actually based on - querying it gives a real number
instead of a guess.

Requires the service account to have the "Monitoring Viewer" IAM role
(read-only, granted once in Cloud Console - not something this app can grant
itself). Does not touch Vision API quota or cost - Cloud Monitoring is a
separate service with its own free-tier read access.
"""

from __future__ import annotations

import datetime as dt
import json


def get_vision_api_request_count_this_month(credentials_path: str) -> int:
    """Returns the real request count Google recorded for the Vision API,
    summed over the current calendar month (UTC) to today. Requires the
    Monitoring Viewer IAM role on the service account backing
    credentials_path - raises on missing permission rather than silently
    returning 0, so a misconfiguration is visible, not mistaken for "no
    usage yet"."""
    from google.cloud import monitoring_v3
    from google.oauth2 import service_account

    with open(credentials_path, encoding="utf-8") as f:
        project_id = json.load(f)["project_id"]

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = monitoring_v3.MetricServiceClient(credentials=credentials)

    now = dt.datetime.now(dt.timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    interval = monitoring_v3.TimeInterval(start_time=month_start, end_time=now)
    aggregation = monitoring_v3.Aggregation(
        alignment_period={"seconds": int((now - month_start).total_seconds()) or 1},
        per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_SUM,
    )

    request = monitoring_v3.ListTimeSeriesRequest(
        name=f"projects/{project_id}",
        filter=(
            'metric.type="serviceruntime.googleapis.com/api/request_count" '
            'AND resource.label.service="vision.googleapis.com"'
        ),
        interval=interval,
        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        aggregation=aggregation,
    )

    total = 0
    for series in client.list_time_series(request=request):
        for point in series.points:
            total += int(point.value.int64_value)
    return total
