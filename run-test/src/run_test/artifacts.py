"""OBS bundle names observed in the live 7day-cache listing, without guessing identity."""
import re


def wheel_identity(url):
    name = url.rsplit('/', 1)[-1]
    old = re.fullmatch(r'wheels-([0-9a-f]{10})\.tar\.gz', name)
    if old:
        return {'pytorch_short': old[1], 'format': 'legacy'}
    new = re.fullmatch(r'wheels-([0-9a-f]{10})-npu-([0-9a-f]{10})-py(\d+)-(.+)-(aarch64|x86_64)(?:-v\d+)?\.tar\.gz', name)
    if new:
        return {'pytorch_short':new[1], 'torch_npu_short':new[2], 'python_tag':new[3],
                'build_environment':new[4], 'architecture':new[5], 'format':'qualified'}
    # wheels-torchnpu-* is a different product, not a full PyTorch + torch-npu bundle.
    return None


def select_wheel(listing, bucket, sha10, config):
    required = (config.get('obs') or {}).get('wheel_name_contains') or []
    if isinstance(required, str):
        required = [required]
    candidates = []
    for url in listing:
        identity = wheel_identity(url)
        if (url.startswith(bucket + '/') and identity and identity['pytorch_short'] == sha10
                and all(token in url.rsplit('/', 1)[-1] for token in required)):
            candidates.append(url)
    if len(candidates) > 1:
        raise ValueError(f'{sha10}: 多个 wheel 构建匹配；请设置 obs.wheel_name_contains: {sorted(candidates)}')
    return candidates[0] if candidates else None


def inventory(listing, bucket, config):
    wheels = {u:wheel_identity(u) for u in listing if wheel_identity(u)}
    groups = sorted({i['pytorch_short'] for i in wheels.values()})
    pairs, ambiguous = [], []
    for sha in groups:
        source = f'{bucket}/pytorch-source-{sha}.tar.gz'
        if source not in listing:
            continue
        try:
            wheel = select_wheel(listing, bucket, sha, config)
        except ValueError as error:
            ambiguous.append(str(error)); continue
        if wheel:
            pairs.append({'source':source, 'wheels':wheel, **wheels[wheel]})
    return {'objects':len(listing), 'full_wheel_bundles':len(wheels), 'unique_pairs':pairs,
            'ambiguous':ambiguous,
            'identity_limit':'Filename pairing only; commit must be resolved on upstream main and wheel contents verified before execution.'}
