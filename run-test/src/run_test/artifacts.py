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
    # 2026-10-22 起构建方把全量 wheels 包拆成两个对象（均带 -vN 后缀）：
    # wheels-pytorch-{sha}-…（torch + torchvision）与
    # wheels-torchnpu-{npu_sha}-{sha}-…（torch_npu）。实证：2026-10-26 下载
    # b43489ea06 / 020ab1c835 配对包核对 tar 内容。
    split_pt = re.fullmatch(r'wheels-pytorch-([0-9a-f]{10})-py(\d+)-(.+)-(aarch64|x86_64)(?:-v\d+)?\.tar\.gz', name)
    if split_pt:
        return {'pytorch_short':split_pt[1], 'python_tag':split_pt[2],
                'build_environment':split_pt[3], 'architecture':split_pt[4],
                'format':'split-pytorch'}
    # 不带 -vN 后缀的 wheels-torchnpu-* 仍是另一种产品（torch_npu 独立构建），
    # 不是完整配套包，继续不识别（test_execution_policy 固化该边界）。
    split_npu = re.fullmatch(r'wheels-torchnpu-([0-9a-f]{10})-([0-9a-f]{10})-py(\d+)-(.+)-(aarch64|x86_64)-v\d+\.tar\.gz', name)
    if split_npu:
        return {'torch_npu_short':split_npu[1], 'pytorch_short':split_npu[2],
                'python_tag':split_npu[3], 'build_environment':split_npu[4],
                'architecture':split_npu[5], 'format':'split-torchnpu'}
    return None


def _token_filter(config):
    required = (config.get('obs') or {}).get('wheel_name_contains') or []
    if isinstance(required, str):
        required = [required]
    return required


def _match(listing, bucket, sha10, required, formats):
    candidates = []
    for url in listing:
        identity = wheel_identity(url)
        if (url.startswith(bucket + '/') and identity
                and identity['format'] in formats and identity['pytorch_short'] == sha10
                and all(token in url.rsplit('/', 1)[-1] for token in required)):
            candidates.append(url)
    if len(candidates) > 1:
        raise ValueError(f'{sha10}: 多个 wheel 构建匹配；请设置 obs.wheel_name_contains: {sorted(candidates)}')
    return candidates[0] if candidates else None


def select_bundle(listing, bucket, sha10, config):
    """按对象名配对一个 sha 的 wheels 主体与独立 torch_npu 包。

    返回 {'wheels': 主体包 URL 或 None, 'wheels_npu': 拆分包的 torch_npu
    包 URL 或 None}；legacy/qualified 格式 torch_npu 在主体包内，
    wheels_npu 恒为 None。两类候选各自歧义都算构建歧义。
    """
    required = _token_filter(config)
    wheels = _match(listing, bucket, sha10, required,
                    ('legacy', 'qualified', 'split-pytorch'))
    wheels_npu = _match(listing, bucket, sha10, required, ('split-torchnpu',))
    return {'wheels': wheels, 'wheels_npu': wheels_npu}


def select_wheel(listing, bucket, sha10, config):
    return select_bundle(listing, bucket, sha10, config)['wheels']


def inventory(listing, bucket, config):
    wheels = {u:wheel_identity(u) for u in listing if wheel_identity(u)}
    groups = sorted({i['pytorch_short'] for i in wheels.values()})
    pairs, ambiguous = [], []
    for sha in groups:
        source = f'{bucket}/pytorch-source-{sha}.tar.gz'
        if source not in listing:
            continue
        try:
            bundle = select_bundle(listing, bucket, sha, config)
        except ValueError as error:
            ambiguous.append(str(error)); continue
        if bundle['wheels']:
            pairs.append({'source':source, 'wheels':bundle['wheels'],
                          'wheels_npu':bundle['wheels_npu'], **wheels[bundle['wheels']]})
    return {'objects':len(listing), 'full_wheel_bundles':len(wheels), 'unique_pairs':pairs,
            'ambiguous':ambiguous,
            'identity_limit':'Filename pairing only; commit must be resolved on upstream main and wheel contents verified before execution.'}
