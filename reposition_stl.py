# ============================================================
#  reposition_stl.py
#  STL 再配置 + Surface Deform 支持体メッシュ + 曲面刻印
#  Blender 4.5 LTS 用
# ============================================================

import bpy
import sys
from pathlib import Path
from datetime import datetime
from mathutils import Vector, Matrix
import math
import re

# ------------------------------------------------------------
# 刻印パラメータ
# ------------------------------------------------------------

# -X 側（テキストA）
TEXT_A_Y_OFFSET = -6.0      # 平坦帯中心からのYオフセット(mm)
TEXT_A_Z_OFFSET = 8.0       # 底面(Z=0)からの高さ(mm)

# +X 側（テキストB）
TEXT_B_Y_OFFSET = 5.0       # 平坦帯中心からのYオフセット(mm)
TEXT_B_Z_OFFSET = 7.0       # 底面(Z=0)からの高さ(mm)

TEXT_HEIGHT_MM = 12.0       # 文字高さ（基準）
TEXT_DEPTH_MM  = 1.5        # 凹み深さ（Text extrude)

# 側壁平坦帯検出用
SIDE_SCAN_WIDTH   = 1.0     # 側壁から±1.0mm 以内の頂点だけを使用
FLAT_BINS_Y       = 24      # Y方向の分割数
MIN_VERTS_PER_BIN = 10      # 1ビンに最低何頂点あれば有効とみなすか

# 支持体メッシュ（グリッド平面）の分割
PLANE_SUBDIV_X = 24        # Z方向分割
PLANE_SUBDIV_Y = 12        # Y方向分割

# 支持体を側壁からどれだけ外側に置くか
SIDE_OUT_OFFSET = 2.0       # mm

# ------------------------------------------------------------
# ログユーティリティ
# ------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


# ------------------------------------------------------------
# コマンドライン引数解析
#   blender -b -P reposition_stl.py -- --rot-angle=30 <dir1> <dir2> ...
# ------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" not in argv:
        return {"dirs": [], "rot_angle": 0.0}

    idx  = argv.index("--")
    args = argv[idx + 1:]

    input_dirs = []
    rot_angle  = 0.0

    for a in args:
        if a.startswith("--rot-angle="):
            try:
                rot_angle = float(a.split("=", 1)[1])
            except Exception:
                warn(f"rot-angle 解析失敗: {a}")
            continue

        p = Path(a).resolve()
        if p.is_dir():
            input_dirs.append(p)
        else:
            warn(f"入力パスがフォルダではありません: {p}")

    return {"dirs": input_dirs, "rot_angle": rot_angle}


# ------------------------------------------------------------
# シーン初期化
# ------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.view_layer.update()


# ------------------------------------------------------------
# STL インポート / エクスポート
# ------------------------------------------------------------

def import_stl(filepath: Path):
    log(f"  ▶ STLインポート: {filepath}")
    bpy.ops.wm.stl_import(filepath=str(filepath))
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    if not meshes:
        warn("    メッシュオブジェクトが見つかりませんでした。")
    else:
        log(f"    インポートされたメッシュ数: {len(meshes)}")
    return meshes


def export_stl(obj, out_path: Path):
    log(f"  ▶ STLエクスポート: {out_path}")
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(filepath=str(out_path))


# ------------------------------------------------------------
# メッシュ結合 / 再配置
# ------------------------------------------------------------

def join_meshes(meshes):
    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]

    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    log(f"    メッシュを結合しました: {joined.name}")
    return joined


def recenter_and_orient(obj):
    # スケール適用
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    verts = obj.data.vertices
    if not verts:
        warn("    頂点が存在しません。スキップします。")
        return

    xs = [v.co.x for v in verts]
    ys = [v.co.y for v in verts]
    zs = [v.co.z for v in verts]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5

    translation = Vector((-center_x, -center_y, -min_z))
    log(f"    平行移動ベクトル: {translation}")

    obj.data.transform(Matrix.Translation(translation))
    obj.location = (0.0, 0.0, 0.0)
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    log(f"    再配置後: dims={obj.dimensions}, loc={obj.location}, rot={obj.rotation_euler}")


# ------------------------------------------------------------
# 出力フォルダ
# ------------------------------------------------------------

def ensure_reposition_output_dir(input_dir: Path) -> Path:
    parent = input_dir.parent
    mmdd   = datetime.now().strftime("%m%d")
    out_dir = parent / f"{mmdd} 再配置後stl"
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"出力フォルダを作成しました: {out_dir}")
    else:
        log(f"出力フォルダを使用します: {out_dir}")
    return out_dir


def ensure_labeled_output_dir(input_dir: Path) -> Path:
    parent = input_dir.parent
    mmdd   = datetime.now().strftime("%m%d")
    out_dir = parent / f"{mmdd} stl刻印あり"
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"刻印用出力フォルダを作成しました: {out_dir}")
    else:
        log(f"刻印用出力フォルダを使用します: {out_dir}")
    return out_dir


# ------------------------------------------------------------
# ファイル名 → 刻印テキスト解析
# ------------------------------------------------------------

def parse_labels_from_filename(stl_path: Path):
    """
    ファイル名: テキストA_テキストB_uまたはl[NN].stl
    例: Cuore0816_TagawaNoa-F_u11.stl
       -> text_a = 'Cuore0816'
          text_b_full = 'TagawaNoa-F_u'
    """
    stem  = stl_path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        warn(f"  ファイル名の形式が想定外です（'A_B_uNN' ではない）: {stem}")
        return None

    text_a = parts[0]
    text_b = parts[1]
    last   = parts[2]

    m = re.fullmatch(r"([ulUL])(\d{0,2})?", last)
    if not m:
        warn(f"  u/l[NN] 部分が解析できません: {last}")
        return None

    jaw = m.group(1).lower()  # 'u' または 'l'
    text_b_full = f"{text_b}_{jaw}"
    return text_a, text_b_full


# ------------------------------------------------------------
# バウンディングボックス取得
# ------------------------------------------------------------

def get_bounds(obj):
    verts = obj.data.vertices
    xs = [v.co.x for v in verts]
    ys = [v.co.y for v in verts]
    zs = [v.co.z for v in verts]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center_y = (min_y + max_y) * 0.5

    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_z": min_z,
        "max_z": max_z,
        "center_y": center_y,
    }


# ------------------------------------------------------------
# 側壁の「平らなY帯」を検出
# ------------------------------------------------------------

def find_flat_band(obj, side: str):
    """
    側壁のうち、最もZ変動が少ない「平坦なY帯」を検出する。
    戻り値: { "y_min": .., "y_max": .., "y_center": .. }
    """
    verts = obj.data.vertices
    xs = [v.co.x for v in verts]
    min_x, max_x = min(xs), max(xs)

    target_x = min_x if side == "NEG_X" else max_x

    # 側壁近傍の頂点のみ抽出
    near = [v.co for v in verts if abs(v.co.x - target_x) <= SIDE_SCAN_WIDTH]
    if not near:
        b = get_bounds(obj)
        return {
            "y_min": b["min_y"],
            "y_max": b["max_y"],
            "y_center": b["center_y"],
        }

    ys = [p.y for p in near]
    min_y, max_y = min(ys), max(ys)
    span_y = max_y - min_y
    if span_y < 1e-6:
        y_center = 0.5 * (min_y + max_y)
        return {"y_min": min_y, "y_max": max_y, "y_center": y_center}

    # Y方向を FLAT_BINS_Y 個に分割し、各帯でZ変動を見る
    bin_min_z = [float("inf")]   * FLAT_BINS_Y
    bin_max_z = [float("-inf")]  * FLAT_BINS_Y
    bin_min_y = [float("inf")]   * FLAT_BINS_Y
    bin_max_y = [float("-inf")]  * FLAT_BINS_Y
    bin_count = [0]              * FLAT_BINS_Y

    for p in near:
        idx = int((p.y - min_y) / span_y * FLAT_BINS_Y)
        if idx < 0:
            idx = 0
        elif idx >= FLAT_BINS_Y:
            idx = FLAT_BINS_Y - 1

        bin_count[idx] += 1
        if p.z < bin_min_z[idx]:
            bin_min_z[idx] = p.z
        if p.z > bin_max_z[idx]:
            bin_max_z[idx] = p.z
        if p.y < bin_min_y[idx]:
            bin_min_y[idx] = p.y
        if p.y > bin_max_y[idx]:
            bin_max_y[idx] = p.y

    best_idx   = None
    best_score = None

    for i in range(FLAT_BINS_Y):
        if bin_count[i] < MIN_VERTS_PER_BIN:
            continue
        z_range = bin_max_z[i] - bin_min_z[i]
        if z_range < 0:
            continue
        if best_score is None or z_range < best_score:
            best_score = z_range
            best_idx   = i

    if best_idx is None:
        # 十分な頂点がない場合は側壁全体を使う
        y_center = 0.5 * (min_y + max_y)
        return {"y_min": min_y, "y_max": max_y, "y_center": y_center}

    y_min_band = bin_min_y[best_idx]
    y_max_band = bin_max_y[best_idx]
    if y_min_band == float("inf") or y_max_band == float("-inf"):
        y_min_band, y_max_band = min_y, max_y

    y_center = 0.5 * (y_min_band + y_max_band)

    return {"y_min": y_min_band, "y_max": y_max_band, "y_center": y_center}


# ------------------------------------------------------------
# 支持体グリッド平面の生成（Surface Deform ターゲット）
# ------------------------------------------------------------

def create_support_plane(base_obj, side: str, band: dict, bounds: dict, text_half_y: float, target_z: float):
    """
    歯列側壁の前に、Y-Z 面に平行なグリッド平面を生成する。
    この平面に Text を Surface Deform でバインドし、
    平面を Shrinkwrap(Project) で曲面に沿わせて、文字を曲面に追従させる。
    """

    band_y_min   = band["y_min"]
    band_y_max   = band["y_max"]
    band_center  = band["y_center"]
    band_height  = max(band_y_max - band_y_min, 1e-3)

    # 支持体平面のY方向半径（文字の高さに余裕を持たせる）
    plane_half_y = max(band_height * 0.6, text_half_y * 1.4)
    # Z方向半径（上下方向。とりあえず文字高さを基準）
    plane_half_z = max(TEXT_HEIGHT_MM * 0.7, 6.0)

    # 側壁から少し外側へ
    if side == "NEG_X":
        x_pos = bounds["min_x"] - SIDE_OUT_OFFSET
        rot_y = math.radians(90.0)   # 法線 -X向き
    else:
        x_pos = bounds["max_x"] + SIDE_OUT_OFFSET
        rot_y = math.radians(-90.0)  # 法線 +X向き

    # 支持体平面生成（Grid）
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=PLANE_SUBDIV_X,
        y_subdivisions=PLANE_SUBDIV_Y,
        size=1.0,
        location=(0.0, 0.0, 0.0)
    )
    plane = bpy.context.object

    # XY平面 → YZ平面へ回転
    plane.rotation_euler = (0.0, rot_y, 0.0)
    bpy.context.view_layer.update()

    # ローカル X → 世界Z, ローカルY → 世界Y になる前提でスケール
    plane.scale = (plane_half_z, plane_half_y, 1.0)
    bpy.context.view_layer.update()

    # スケール・回転を適用
    bpy.ops.object.select_all(action='DESELECT')
    plane.select_set(True)
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # 位置合わせ（中心を band_center + offset, 指定Zへ）
    plane.location = (x_pos, band_center, target_z)
    bpy.context.view_layer.update()

    return plane


# ------------------------------------------------------------
# Surface Deform + Shrinkwrap を用いて曲面に沿わせた刻印用メッシュを生成
# ------------------------------------------------------------

def create_surface_deform_text_cutter(base_obj, text_body: str, side: str, band: dict, bounds: dict,
                                      base_y_offset: float, base_z_offset: float):
    """
    - Text を生成（文字の縦方向を Y軸に平行）
    - 側壁の平坦帯と文字幅から Y位置を決める
    - 支持体グリッド平面を生成
    - Text を Mesh に変換し、Surface Deform で支持体平面にバインド
    - 支持体平面を Shrinkwrap(Project) で歯列曲面に沿わせる
    - Text の Surface Deform を適用して、曲面に沿った文字メッシュを返す
    """
    band_y_min = band["y_min"]
    band_y_max = band["y_max"]
    band_center = band["y_center"]
    band_height = max(band_y_max - band_y_min, 1e-3)

    # 1) Text 生成（原点）
    bpy.ops.object.text_add(location=(0.0, 0.0, 0.0))
    text_obj = bpy.context.object
    text_obj.data.body    = text_body
    text_obj.data.extrude = TEXT_DEPTH_MM

    # 文字の縦(Z)を Y軸に揃えつつ、面法線を ±X へ
    if side == "NEG_X":
        text_obj.rotation_euler = (
            math.radians(90.0),  # X回転
            0.0,
            math.radians(180.0)
        )
    else:
        text_obj.rotation_euler = (
            math.radians(90.0),
            0.0,
            0.0
        )

    bpy.context.view_layer.update()

    # 高さを TEXT_HEIGHT_MM に揃える（dimensions.z を基準に）
    cur_h = text_obj.dimensions.z
    if cur_h > 0:
        s = TEXT_HEIGHT_MM / cur_h
        text_obj.scale = (s, s, s)
        bpy.context.view_layer.update()

    # 文字幅(世界Y方向)の半分
    dims = text_obj.dimensions
    text_half_y = 0.5 * dims.y

    # 平坦帯の中に文字が収まるよう、Y位置を決定
    margin = 0.2
    low_limit  = band_y_min + text_half_y + margin
    high_limit = band_y_max - text_half_y - margin

    base_y = band_center + base_y_offset
    if high_limit > low_limit:
        target_y = max(low_limit, min(base_y, high_limit))
    else:
        target_y = base_y

    target_z = base_z_offset

    # 支持体平面を生成（位置は band_center だが、Y方向に十分なサイズを持つ）
    support_plane = create_support_plane(
        base_obj,
        side=side,
        band=band,
        bounds=bounds,
        text_half_y=text_half_y,
        target_z=target_z
    )

    # Text を支持体平面上の適切な位置へ移動（X は支持体と合わせる）
    if side == "NEG_X":
        x_pos = bounds["min_x"] - SIDE_OUT_OFFSET
    else:
        x_pos = bounds["max_x"] + SIDE_OUT_OFFSET

    text_obj.location = (x_pos, target_y, target_z)
    bpy.context.view_layer.update()

    # 2) Text を Mesh に変換
    bpy.ops.object.select_all(action='DESELECT')
    text_obj.select_set(True)
    bpy.context.view_layer.objects.active = text_obj
    bpy.ops.object.convert(target='MESH')
    cutter_obj = bpy.context.object

    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.context.view_layer.update()

    # 3) Text に Surface Deform モディファイアを追加し、支持体にバインド
    surf_mod = cutter_obj.modifiers.new(name="SurfDef", type='SURFACE_DEFORM')
    surf_mod.target = support_plane

    bpy.ops.object.select_all(action='DESELECT')
    cutter_obj.select_set(True)
    bpy.context.view_layer.objects.active = cutter_obj
    bpy.ops.object.surfacedeform_bind(modifier=surf_mod.name)

    # 4) 支持体平面に Shrinkwrap(Project) を追加して歯列に沿わせる
    shrink = support_plane.modifiers.new(name="SideWrap", type='SHRINKWRAP')
    shrink.target = base_obj
    shrink.wrap_method = 'PROJECT'
    shrink.use_project_x = True
    shrink.use_project_y = False
    shrink.use_project_z = False
    if side == "NEG_X":
        shrink.use_negative_direction = False
        shrink.use_positive_direction = True   # +X 方向へ投影
    else:
        shrink.use_negative_direction = True   # -X 方向へ投影
        shrink.use_positive_direction = False
    shrink.offset = 0.0

    bpy.ops.object.select_all(action='DESELECT')
    support_plane.select_set(True)
    bpy.context.view_layer.objects.active = support_plane
    bpy.ops.object.modifier_apply(modifier=shrink.name)
    bpy.context.view_layer.update()

    # 5) Text側の Surface Deform を適用（曲面に沿った文字メッシュになる）
    bpy.ops.object.select_all(action='DESELECT')
    cutter_obj.select_set(True)
    bpy.context.view_layer.objects.active = cutter_obj
    bpy.ops.object.modifier_apply(modifier=surf_mod.name)
    bpy.context.view_layer.update()

    # 支持体平面は不要なので削除
    bpy.data.objects.remove(support_plane, do_unlink=True)

    return cutter_obj


# ------------------------------------------------------------
# ブーリアン適用
# ------------------------------------------------------------

def apply_boolean_engrave(target_obj, cutter_obj, label: str):
    mod = target_obj.modifiers.new(name=f"Engrave_{label}", type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver    = 'EXACT'
    mod.object    = cutter_obj

    bpy.ops.object.select_all(action='DESELECT')
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    bpy.data.objects.remove(cutter_obj, do_unlink=True)


# ------------------------------------------------------------
# 刻印処理
# ------------------------------------------------------------

def engrave_labels(obj, text_a: str, text_b: str):
    verts = obj.data.vertices
    if not verts:
        warn("  刻印対象オブジェクトに頂点がありません。")
        return

    bounds   = get_bounds(obj)
    band_neg = find_flat_band(obj, "NEG_X")
    band_pos = find_flat_band(obj, "POS_X")

    # -X側：テキストA
    cutter_a = create_surface_deform_text_cutter(
        base_obj=obj,
        text_body=text_a,
        side="NEG_X",
        band=band_neg,
        bounds=bounds,
        base_y_offset=TEXT_A_Y_OFFSET,
        base_z_offset=TEXT_A_Z_OFFSET
    )
    apply_boolean_engrave(obj, cutter_a, "A")

    # +X側：テキストB
    cutter_b = create_surface_deform_text_cutter(
        base_obj=obj,
        text_body=text_b,
        side="POS_X",
        band=band_pos,
        bounds=bounds,
        base_y_offset=TEXT_B_Y_OFFSET,
        base_z_offset=TEXT_B_Z_OFFSET
    )
    apply_boolean_engrave(obj, cutter_b, "B")


# ------------------------------------------------------------
# 1 STL ファイル処理
# ------------------------------------------------------------

def process_stl_file(stl_path: Path, out_dir_repos: Path, out_dir_label: Path, rot_angle: float):
    log("========================================")
    log(f"STL処理開始: {stl_path}")

    reset_scene()

    meshes = import_stl(stl_path)
    if not meshes:
        warn("  メッシュが無いためスキップします。")
        return

    obj = join_meshes(meshes)
    if obj is None:
        warn("  結合結果オブジェクトが無いためスキップします。")
        return

    recenter_and_orient(obj)

    # 再配置後 STL を出力
    out_path_repos = out_dir_repos / stl_path.name
    export_stl(obj, out_path_repos)

    # ファイル名から刻印テキスト取得
    labels = parse_labels_from_filename(stl_path)
    if labels is None:
        warn("  刻印テキストが取得できないため、刻印処理をスキップします。")
        log(f"STL処理完了: {stl_path}")
        return

    text_a, text_b = labels
    log(f"  刻印テキストA: {text_a}")
    log(f"  刻印テキストB: {text_b}")

    # XY 平面回転（Z軸回り）を刻印前に適用
    if rot_angle != 0.0:
        log(f"  ▶ XY平面回転を適用: {rot_angle}°")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        obj.rotation_euler[2] += math.radians(rot_angle)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        bpy.context.view_layer.update()

    # Surface Deform + 支持体メッシュで曲面に沿った刻印
    engrave_labels(obj, text_a, text_b)

    # 刻印あり STL を出力
    labeled_name = f"{stl_path.stem}(刻印あり){stl_path.suffix}"
    out_path_label = out_dir_label / labeled_name
    export_stl(obj, out_path_label)

    log(f"STL処理完了: {stl_path}")


# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------

def main():
    args       = parse_args()
    input_dirs = args["dirs"]
    rot_angle  = args["rot_angle"]

    if not input_dirs:
        error("入力フォルダが指定されていません。 '--' の後に1つ以上のフォルダを指定してください。")
        return

    log("============================================")
    log("  STL 再配置＋曲面刻印プログラム (Blender 4.5)")
    log("============================================")
    log(f"入力フォルダ数: {len(input_dirs)}")
    log(f"XY平面回転角度: {rot_angle}°")

    for in_dir in input_dirs:
        log("--------------------------------------------")
        log(f"入力フォルダ: {in_dir}")

        if not in_dir.is_dir():
            warn(f"フォルダではありません。スキップ: {in_dir}")
            continue

        out_dir_repos  = ensure_reposition_output_dir(in_dir)
        out_dir_label  = ensure_labeled_output_dir(in_dir)

        stl_files = sorted(
            p for p in in_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".stl"
        )

        if not stl_files:
            warn("  対象STLが見つかりません。")
            continue

        log(f"  対象STL数: {len(stl_files)}")

        for stl_path in stl_files:
            process_stl_file(stl_path, out_dir_repos, out_dir_label, rot_angle)

    log("全処理が完了しました。")


if __name__ == "__main__":
    main()
