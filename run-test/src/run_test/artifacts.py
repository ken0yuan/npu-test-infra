"""OBS bundle names observed in the live 7day-cache listing, without guessing identity."""
import re
from datetime import datetime


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


def _matches(listing, bucket, sha10, required, formats):
    candidates = []
    for url in listing:
        identity = wheel_identity(url)
        if (url.startswith(bucket + '/') and identity
                and identity['format'] in formats and identity['pytorch_short'] == sha10
                and all(token in url.rsplit('/', 1)[-1] for token in required)):
            candidates.append(url)
    return candidates


def _pair_key(url):
    identity = wheel_identity(url)
    version = re.search(r'-v(\d+)\.tar\.gz$', url)
    return (identity['python_tag'], identity['architecture'],
            identity['build_environment'], version[1] if version else None)


def _latest(listing, sha10, bundles):
    if len(bundles) <= 1:
        return bundles[0] if bundles else (None, None)
    ranked = []
    for bundle in bundles:
        times = []
        for url in filter(None, bundle):
            value = listing[url].get('last_modified')
            try:
                stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
                if stamp.tzinfo is None:
                    raise ValueError('timezone missing')
            except (AttributeError, TypeError, ValueError):
                raise ValueError(
                    f'{sha10}: 多个 wheel 构建缺少有效的 OBS last_modified；'
                    f'无法选择最新完整套包，请设置 obs.wheel_name_contains: {url}'
                ) from None
            times.append(stamp.timestamp())
        # A split bundle becomes complete when its last member is uploaded.
        # Equal timestamps use member times and URLs for stable listing order.
        ranked.append(((max(times), tuple(times), tuple(u or '' for u in bundle)), bundle))
    return max(ranked, key=lambda item: item[0])[1]


def select_bundle(listing, bucket, sha10, config):
    """选择同一 sha、配置过滤范围内最新上传的完整 wheels 套包。

    返回 {'wheels': 主体包 URL 或 None, 'wheels_npu': 拆分包的 torch_npu
    包 URL 或 None}；legacy/qualified 格式 torch_npu 在主体包内，
    wheels_npu 恒为 None。拆分包必须匹配 Python、架构、构建环境和包格式
    版本。按套包最后一个成员的 OBS last_modified 排序，未上传完整的新
    构建不挤掉已有完整套包；多个候选缺少可信时间戳时保留显式歧义。
    """
    required = _token_filter(config)
    wheels = _matches(listing, bucket, sha10, required,
                      ('legacy', 'qualified', 'split-pytorch'))
    npu = _matches(listing, bucket, sha10, required, ('split-torchnpu',))
    complete, incomplete = [], []
    for wheel in wheels:
        if wheel_identity(wheel)['format'] != 'split-pytorch':
            complete.append((wheel, None))
            continue
        paired = [url for url in npu if _pair_key(url) == _pair_key(wheel)]
        if paired:
            complete.extend((wheel, url) for url in paired)
        else:
            incomplete.append((wheel, None))
    wheel, torch_npu = _latest(listing, sha10, complete or incomplete)
    return {'wheels': wheel, 'wheels_npu': torch_npu}


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
