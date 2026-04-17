"""
utils/logger.py
---------------
Structured logging and CloudWatch custom metrics for the NRFI pipeline.

log()    — Emits JSON-structured lines to stdout.
           SageMaker and Lambda both stream stdout to CloudWatch Logs automatically.

metric() — Emits a CloudWatch custom metric under the NRFI/Pipeline namespace.
           Fails silently — a metric failure never crashes the pipeline.
"""

import json
from datetime import datetime, timezone

CW_NAMESPACE = 'NRFI/Pipeline'
_cw_client = None


def _get_cw():
    global _cw_client
    if _cw_client is None:
        try:
            import boto3
            _cw_client = boto3.client('cloudwatch', region_name='us-east-1')
        except Exception:
            pass
    return _cw_client


def log(level, msg, **ctx):
    """
    Print a structured JSON log line to stdout.

    Args:
        level: 'INFO' | 'WARN' | 'ERROR'
        msg:   Human-readable message string
        **ctx: Additional key=value context fields
    """
    rec = {
        'level': level,
        'ts':    datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'msg':   msg,
    }
    rec.update(ctx)
    print(json.dumps(rec, default=str), flush=True)


def metric(name, value, unit='Count', dimensions=None):
    """
    Emit a CloudWatch custom metric under the NRFI/Pipeline namespace.

    Uses daily StorageResolution (86400s) to minimize cost.
    PutMetricData does not support resource-level IAM restrictions — the
    SageMaker/Lambda role must have cloudwatch:PutMetricData on "*".

    Args:
        name:       Metric name (e.g. 'LRPickCount')
        value:      Numeric value
        unit:       CloudWatch unit string (default 'Count')
                    Common: 'Count', 'None', 'Dollars', 'Percent'
        dimensions: Optional dict of {Name: Value} dimension pairs
                    (e.g. {'Model': 'LR'})

    Fails silently — never crashes the pipeline.
    """
    cw = _get_cw()
    if cw is None:
        return
    try:
        dims = [{'Name': k, 'Value': str(v)} for k, v in (dimensions or {}).items()]
        cw.put_metric_data(
            Namespace=CW_NAMESPACE,
            MetricData=[{
                'MetricName':       name,
                'Value':            float(value),
                'Unit':             unit,
                'Dimensions':       dims,
                'Timestamp':        datetime.now(timezone.utc),
                'StorageResolution': 86400,  # daily resolution — lowest cost tier
            }]
        )
    except Exception as ex:
        # Log warning to stdout but never raise — metrics are non-critical
        log('WARN', f'CloudWatch metric failed: {name}={value}', error=str(ex))
