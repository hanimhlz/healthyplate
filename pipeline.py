import re
from flask import logging
import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ===== KONSTANTA =====
fitur_cols = [
    "Kalori (Kkal)", "Protein (G)", "Lemak (G)", "Karbohidrat (G)",
    "Serat (G)", "Kalium (Mg)", "Natrium (Mg)", "Gula (G)", "Kolesterol (Mg)"
]

NUTRISI_COLS = fitur_cols

weights = np.array([
    0.059716, 0.114206, 0.144458, 0.054114,
    0.029929, 0.035556, 0.299280, 0.077812, 0.185529
])
weights = weights / weights.sum()

komponen_map = {
    "Kalori (Kkal)":   "Energi (kkal)",
    "Protein (G)":     "Protein (g)",
    "Lemak (G)":       "Lemak (g)",
    "Karbohidrat (G)": "Karbohidrat (g)",
    "Serat (G)":       "Serat (g)",
    "Kalium (Mg)":     "Kalium (mg)",
    "Natrium (Mg)":    "Natrium (mg)",
    "Gula (G)":        "Gula (g)",
    "Kolesterol (Mg)": "Kolesterol (mg)"
}

NAMA_HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

# ===== KEYWORD MAKANAN PAGI =====
KEYWORD_PAGI = [
    "bubur", "oatmeal", "oat", "pancake", "roti", "sereal", "granola",
    "telur", "omelet", "sandwich", "toast", "muffin", "smoothie bowl",
    "sup", "soto", "pecel", "lontong", "ketupat", "nasi uduk",
    "nasi kuning", "nasi goreng", "mi goreng", "mi kuah", "bihun goreng",
    "kwetiau", "cap cay", "gado-gado", "ubi rebus", "singkong rebus",
    "jagung rebus", "kue basah", "lemper", "risoles", "combro",
    "misro", "cireng", "cimol", "batagor", "siomay", "dimsum",
    "hakau", "shumai", "bakpao", "areh-arem", "mendut",
    "klepon", "onde-onde", "pisang goreng", "pisang rebus",
    "talas", "ganyong", "suweg", "gembili"
]

KEYWORD_NON_PAGI = [
    "sate", "rendang", "gulai", "tongseng", "rawon",
    "steak", "ayam bakar", "ikan bakar", "bebek goreng",
    "mie instan", "martabak manis", "seafood", "lalapan",
    "sambal matah", "rica", "balado", "kari", "katsu",
    "coto", "pallubasa", "konro", "empal gentong",
    "nasi padang", "nasi kapau", "nasi liwet solo"
]

# ===== BLACKLIST FASTFOOD & SNACK TIDAK SEHAT =====
BLACKLIST = [
    "mcd", "burger king", "kfc", "pizza hut", "richeese", "yoshinoya",
    "hokben", "lawson", "pop mie", "indomie", "sedaap", "supermi",
    "pilus", "chitato", "chiki", "taro net", "tic tac", "corntoz",
    "thalasa", "coffee float", "mcflurry", "mcnugget", "mcspicy",
    "sweet and sour sauce", "iced coffee (mcd)", "chocolate sundae",
    "ice cream cone (mcd)", "sausage patty (mcd)", "garuda", "rumput laut kering panggang",
]

# ===== KEYWORD MAKANAN YANG TIDAK PERLU NASI =====
KEYWORD_NO_RICE = [
    "bubur", "lontong", "nasi", "mie", "mi", "pasta", "kentang",
    "oat", "roti", "sereal", "ketan", "ubi", "singkong", "jagung",
    "kwetiau", "bihun", "laksa", "ketupat", "pizza", "sandwich",
    "makaroni", "spageti", "fettuccine", "lasagna", "capcay",
    "cap cay", "bakmi", "bakmie", "yamien", "yamin", "ifumie",
    "ifumi", "mie goreng", "mie kuah", "mie rebus", "mie nyemek",
    "nasi goreng", "nasi uduk", "nasi kuning", "nasi liwet",
    "nasi timbel", "nasi campur", "nasi rames", "nasi kotak",
    "nasi bungkus", "nasi hainan", "nasi kebuli", "nasi briyani",
    "nasi mandhi", "nasi madura", "nasi bogana", "nasi megono",
    "nasi jamblang", "nasi kucing", "nasi angkringan", "nasi pecel",
    "nasi gudeg", "nasi rawon", "nasi soto", "nasi tumpeng",
    "nasi ulam", "nasi biryani", "nasi minyak", "nasi lemak",
    "nasi dagang", "nasi kerabu", "nasi ayam", "nasi bebek",
    "nasi ikan", "nasi daging", "nasi kambing", "nasi sapi", "nasi kebuli",
    "nasi babi", "nasi udang", "nasi cumi", "nasi sotong",
    "nasi kepiting", "nasi lobster", "nasi rajungan", "nasi kerang",
    "nasi tiram"
]

# ===== KEYWORD MINUMAN UNTUK SNACK =====
KEYWORD_MINUMAN = [
    "teh", "kopi", "jus", "juice", "smoothie", "shake",
    "es ", "es teh", "es kopi", "es jeruk", "es kelapa",
    "infused water", "lemonade", "yogurt drink", "susu",
    "wedang", "bandrek", "sekoteng", "bajigur", "bir pletok",
    "dawet", "cendol", "es doger", "es campur", "es buah",
    "air kelapa", "kelapa muda", "sari buah", "sari kedelai",
    "soya", "soy milk", "almond milk", "oat milk",
    "tea", "coffee", "latte", "cappuccino", "americano",
    "mocha", "matcha", "bubble tea", "boba", "thai tea",
    "green tea", "black tea", "jasmine tea", "oolong",
    "herbal tea", "chamomile", "ginger tea", "lemon tea",
    "ice tea", "iced tea", "cold brew", "frappe",
    "protein shake", "whey", "ensure", "ensuren",
    "sari kacang", "susu kedelai", "susu almond", "susu oat",
    "yoghurt", "yogurt", "yakult", "yoghurt drink",
    "es krim",
]


# ===== LOAD DATASET =====
def load_dataset(path="food_datasets.csv"):
    df = pd.read_csv(
        path,
        on_bad_lines='skip',
        engine='python'
    )
    df.columns = df.columns.str.strip().str.title()
    if "Kategori" in df.columns:
        df["Kategori"] = df["Kategori"].astype(str).str.title()
    available = [col for col in NUTRISI_COLS if col in df.columns]
    df[available] = df[available].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=available, inplace=True)
    return df


# ===== AKG =====
def get_akg(age, gender):
    if gender.lower() == "female":
        akg_data = [
            (18, 52, 65, 70, 300, 29, 5000),
            ((19, 29), 55, 60, 65, 360, 32, 4700),
            ((30, 49), 56, 60, 60, 340, 30, 4700),
            ((50, 64), 56, 60, 50, 280, 25, 4700),
            ((65, 80), 53, 58, 45, 230, 22, 4700),
            ("80+", 53, 58, 40, 200, 20, 4700)
        ]
    else:
        akg_data = [
            (18, 60, 75, 85, 400, 37, 5300),
            ((19, 29), 60, 65, 75, 430, 37, 4700),
            ((30, 49), 60, 65, 70, 415, 36, 4700),
            ((50, 64), 60, 65, 60, 340, 30, 4700),
            ((65, 80), 58, 64, 50, 275, 25, 4700),
            ("80+", 58, 64, 45, 235, 22, 4700)
        ]
    for row in akg_data:
        age_group = row[0]
        if isinstance(age_group, tuple):
            if age_group[0] <= age <= age_group[1]:
                return {"bb_akg": row[1], "protein": row[2], "lemak": row[3],
                        "karbo": row[4], "serat": row[5], "kalium": row[6]}
        elif age_group == 18 and age == 18:
            return {"bb_akg": row[1], "protein": row[2], "lemak": row[3],
                    "karbo": row[4], "serat": row[5], "kalium": row[6]}
        elif age_group == "80+" and age >= 80:
            return {"bb_akg": row[1], "protein": row[2], "lemak": row[3],
                    "karbo": row[4], "serat": row[5], "kalium": row[6]}
    return None


# ===== KALKULASI NUTRISI =====
def hitung_bmr(age, gender, weight, height):
    if gender.lower() == "male":
        return (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        return (10 * weight) + (6.25 * height) - (5 * age) - 161


def hitung_tdee(bmr, activity_level):
    activity_dict = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
    return bmr * activity_dict.get(activity_level, 1.2)


def cek_hipertensi(systolic, diastolic):
    if systolic >= 180 or diastolic >= 110:
        return 3
    elif systolic >= 160 or diastolic >= 100:
        return 2
    elif systolic >= 140 or diastolic >= 90:
        return 1
    else:
        return 0


def hitung_batas_nutrisi(tdee, derajat, age, gender, weight):
    akg = get_akg(age, gender)
    if akg is None:
        raise ValueError(f"Umur {age} tidak ditemukan di tabel AKG.")

    koreksi_bb   = weight / akg["bb_akg"]
    meal_share   = 0.8 / 3
    snack_share  = 0.2 / 2
    protein_h    = koreksi_bb * akg["protein"]
    karbo_h      = koreksi_bb * akg["karbo"]
    serat_h      = koreksi_bb * akg["serat"]
    kalium_h     = koreksi_bb * akg["kalium"]
    lemak_h      = (0.25 * tdee) / 9
    natrium_h    = {1: 1200, 2: 800, 3: 400}.get(derajat, 2000)

    return {
        "koreksi_bb":       koreksi_bb,
        "kalori_meal":      tdee * meal_share,
        "kalori_snack":     tdee * snack_share,
        "protein_meal":     protein_h * meal_share,
        "protein_snack":    protein_h * snack_share,
        "karbo_meal":       karbo_h * meal_share,
        "karbo_snack":      karbo_h * snack_share,
        "serat_meal":       serat_h * meal_share,
        "serat_snack":      serat_h * snack_share,
        "kalium_meal":      kalium_h * meal_share,
        "kalium_snack":     kalium_h * snack_share,
        "lemak_meal":       lemak_h * meal_share,
        "lemak_snack":      lemak_h * snack_share,
        "natrium_meal":     natrium_h * meal_share,
        "natrium_snack":    natrium_h * snack_share,
        "gula_meal":        50 * meal_share,
        "gula_snack":       50 * snack_share,
        "kolesterol_meal":  200 * meal_share,
        "kolesterol_snack": 200 * snack_share,
    }


# ===== Rule based filtering =====
def filter_makanan(df, batas, kategori, alergi=None):
    label  = kategori.capitalize()
    suffix = "meal" if kategori.lower() == "meal" else "snack"

    hasil = df[
        (df["Kategori"] == label) &
        (df["Lemak (G)"]       <= batas[f"lemak_{suffix}"]) &
        (df["Natrium (Mg)"]    <= batas[f"natrium_{suffix}"]) &
        (df["Kolesterol (Mg)"] <= batas[f"kolesterol_{suffix}"]) &
        (df["Gula (G)"]        <= batas[f"gula_{suffix}"])
    ].copy()

    nasi_standalone = [
        "nasi putih", "nasi merah", "nasi kuning", "nasi coklat",
        "nasi uduk", "nasi liwet", "nasi jagung", "nasi samin",
        "nasi ketan", "nasi tiwul", "nasi basmati", "nasi gurih",
        "nasi pera", "nasi pulen", "nasi organik", "nasi hitam",
        "nasi cokelat", "nasi beras merah", "nasi beras hitam",
    ]
    hasil = hasil[
        ~hasil["Nama Makanan"].str.strip().str.lower().isin(nasi_standalone)
    ]

    if BLACKLIST:
        blacklist_pattern = "|".join(re.escape(k) for k in BLACKLIST)
        hasil = hasil[~hasil["Nama Makanan"].str.contains(blacklist_pattern, case=False, na=False, regex=True)]

    if alergi:
        allergen_list = [a.strip() for a in str(alergi).split("|") if a.strip()]
        if allergen_list:
            allergen_pattern = "|".join(re.escape(a) for a in allergen_list)
            hasil = hasil[~hasil["Bahan"].str.contains(allergen_pattern, case=False, na=False, regex=True) & ~hasil["Nama Makanan"].str.contains(allergen_pattern, case=False, na=False, regex=True)]

    return hasil.reset_index(drop=True)


# ===== NORMALISASI =====
def build_tabel_nutrisi(batas, tdee, derajat, akg, koreksi_bb):
    natrium_ref = {1: 1200, 2: 800, 3: 400}.get(derajat, 2000)
    return pd.DataFrame({
        "Komponen": [
            "Energi (kkal)", "Protein (g)", "Lemak (g)", "Karbohidrat (g)",
            "Serat (g)", "Kalium (mg)", "Natrium (mg)", "Gula (g)", "Kolesterol (mg)"
        ],
        "Referensi Harian (AKG / Batas)": [
            tdee,
            akg["protein"] * koreksi_bb,
            (0.25 * tdee) / 9,
            akg["karbo"] * koreksi_bb,
            akg["serat"] * koreksi_bb,
            akg["kalium"] * koreksi_bb,
            natrium_ref, 50, 200
        ],
        "Target Makan Utama": [
            batas["kalori_meal"],  batas["protein_meal"],  batas["lemak_meal"],
            batas["karbo_meal"],   batas["serat_meal"],    batas["kalium_meal"],
            batas["natrium_meal"], batas["gula_meal"],     batas["kolesterol_meal"]
        ],
        "Target Camilan": [
            batas["kalori_snack"],  batas["protein_snack"],  batas["lemak_snack"],
            batas["karbo_snack"],   batas["serat_snack"],    batas["kalium_snack"],
            batas["natrium_snack"], batas["gula_snack"],     batas["kolesterol_snack"]
        ]
    })


def akg_normalize(df, daily_reference):
    df_norm = df.copy()
    for i, col in enumerate(fitur_cols):
        df_norm[col] = df_norm[col] / (daily_reference[i] + 1e-9)
    return df_norm


# ===== KNN =====
def weighted_euclidean(user_vector, food_df):
    food_matrix = food_df[fitur_cols].values
    user_vector = np.asarray(user_vector).reshape(1, -1)
    return np.sqrt(np.sum(weights * (food_matrix - user_vector) ** 2, axis=1))


def build_user_vector(tabel_nutrisi, target_col):
    return np.array([
        tabel_nutrisi.loc[tabel_nutrisi["Komponen"] == komponen_map[col], target_col].values[0]
        for col in fitur_cols
    ])


# ===== CLUSTERING  =====
def apply_clustering(data, original_df, batas, kategori, final_k=3):
    # 1. Ambil data kandidat
    top_k_names = data["Nama Makanan"].tolist()
    original_data = original_df[original_df["Nama Makanan"].isin(top_k_names)].copy()
    original_data = original_data.drop_duplicates(subset=["Nama Makanan"]).reset_index(drop=True)

    # Merge dengan Distance dari hasil WED
    original_data = original_data.merge(
        data.drop_duplicates(subset=["Nama Makanan"])[["Nama Makanan", "Distance"]],
        on="Nama Makanan",
        how="left"
    )

    # 2. URUTKAN BERDASARKAN DISTANCE (WED) — ranking tetap dari WED
    original_data = original_data.sort_values("Distance").reset_index(drop=True)
    original_data["Rank"] = original_data.index + 1

    # 3. NORMALISASI 4 FITUR RISIKO PER BATAS AKG dan MINMAX NORMALIZATION
    suffix = "meal" if kategori == "meal" else "snack"
    batas_natrium    = batas[f"natrium_{suffix}"]
    batas_kolesterol = batas[f"kolesterol_{suffix}"]
    batas_lemak      = batas[f"lemak_{suffix}"]
    batas_gula       = batas[f"gula_{suffix}"]

    cluster_data = original_data.copy()
    cluster_data["Natrium_norm"]    = cluster_data["Natrium (Mg)"]    / batas_natrium
    cluster_data["Kolesterol_norm"] = cluster_data["Kolesterol (Mg)"] / batas_kolesterol
    cluster_data["Lemak_norm"]      = cluster_data["Lemak (G)"]       / batas_lemak
    cluster_data["Gula_norm"]       = cluster_data["Gula (G)"]        / batas_gula

    cluster_features_norm = ["Natrium_norm", "Kolesterol_norm", "Lemak_norm", "Gula_norm"]

    scaler = MinMaxScaler()
    cluster_data[cluster_features_norm] = scaler.fit_transform(
        cluster_data[cluster_features_norm]
    )

    # 4. K-MEANS CLUSTERING
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    original_data["Cluster"] = kmeans.fit_predict(cluster_data[cluster_features_norm])

    # 5. GENERATE LABEL DARI CENTROID
    centroid_df = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=cluster_features_norm
    )

    feature_display = {
        "Natrium_norm":    "Natrium",
        "Kolesterol_norm": "Kolesterol",
        "Lemak_norm":      "Lemak",
        "Gula_norm":       "Gula"
    }

    label_profil = {}
    # Hitung rata-rata global tiap fitur di antara semua cluster
    global_mean = centroid_df.mean()  

    for cluster_id in centroid_df.index:
        row = centroid_df.loc[cluster_id]
        
        # 1. Cari fitur mana yang nilainya paling tinggi (dominan) di cluster ini
        dominant_col = row.idxmax()
        dominant_name = feature_display[dominant_col]
        
        # 2. Bandingkan nilai dominan tersebut dengan rata-rata globalnya
        # Jika nilai tertingginya saja masih di bawah rata-rata global, 
        # berarti cluster ini secara umum adalah yang paling rendah risikonya.
        if row[dominant_col] > global_mean[dominant_col]:
            label_profil[cluster_id] = f"Relatif Tinggi {dominant_name}"
        else:
            label_profil[cluster_id] = "Rendah Risiko Relatif"

    # 6. MAP LABEL KE DATAFRAME
    original_data["Profil Risiko"] = original_data["Cluster"].map(label_profil)

    return original_data


# ===== NASI HELPER =====
def is_pagi_food(food_name):
    food_lower = re.sub(r'[^a-z0-9\s]', ' ', str(food_name).lower())
    for kw in KEYWORD_PAGI:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, food_lower):
            for non_kw in KEYWORD_NON_PAGI:
                if re.search(r"\b" + re.escape(non_kw.lower()) + r"\b", food_lower):
                    return False
            return True
    return False


def is_rice_compatible(food_name):
    food_lower = re.sub(r'[^a-z0-9\s]', ' ', str(food_name).lower())
    for kw in KEYWORD_NO_RICE:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, food_lower):
            return False
    return True


def is_beverage(food_name):
    food_lower = re.sub(r'[^a-z0-9\s]', ' ', str(food_name).lower())
    for kw in KEYWORD_MINUMAN:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, food_lower):
            return True
    return False


def get_rice_data(df, rice_option):
    rice_option = str(rice_option).strip().lower()
    rice_map = {
        "nasi_putih": "nasi putih",
        "nasi_merah": "nasi merah",
        "nasi_kuning": "nasi kuning",
        "nasi_uduk": "nasi uduk",
    }
    search_term = rice_map.get(rice_option)
    if search_term is None:
        return None
    rice = df[df["Nama Makanan"].str.strip().str.lower() == search_term]
    return rice.iloc[0] if not rice.empty else None


# ===== MEAL PLAN  =====
def generate_daily_plan(meal_pagi, meal_siang_malam, snack_data,
                        used_meals, used_snacks, tdee, batas,
                        current_day=1, meal_history=None, snack_history=None):
    selected_meals = []
    selected_snacks = []

    is_beverage_mask = snack_data["Nama Makanan"].apply(is_beverage)
    snack_minuman = snack_data[is_beverage_mask]
    snack_padat   = snack_data[~is_beverage_mask]

    def _calc_current_calories():
        total = sum(m["Kalori (Kkal)"].values[0] for m in selected_meals)
        total += sum(s["Kalori (Kkal)"].values[0] for s in selected_snacks)
        return total

    def _normalize_name(name):
        return str(name).strip().lower()

    def _get_blocked(history):
        if history is None:
            return set()
        return {name for name, last_day in history.items() if (current_day - last_day) < 7}

# ===== MEAL PLAN (SEPENUHNYA WED-DRIVEN, TANPA FILTER LABEL) =====
def generate_daily_plan(meal_pagi, meal_siang_malam, snack_data,
                        used_meals, used_snacks, tdee, batas,
                        current_day=1, meal_history=None, snack_history=None):
    selected_meals = []
    selected_snacks = []

    is_beverage_mask = snack_data["Nama Makanan"].apply(is_beverage)
    snack_minuman = snack_data[is_beverage_mask]
    snack_padat   = snack_data[~is_beverage_mask]

    def _calc_current_calories():
        total = sum(m["Kalori (Kkal)"].values[0] for m in selected_meals)
        total += sum(s["Kalori (Kkal)"].values[0] for s in selected_snacks)
        return total

    def _normalize_name(name):
        return str(name).strip().lower()

    def _get_blocked(history):
        if history is None:
            return set()
        return {name for name, last_day in history.items() if (current_day - last_day) < 7}

    def _try_add_from_pool(pool, target_list, used_set, max_count, history=None, top_k=25):
        blocked = _get_blocked(history) | used_set
        blocked |= {_normalize_name(item["Nama Makanan"].values[0]) for item in target_list}

        # Ambil kandidat terbaik berdasarkan JARAK WED (menggunakan top_k)
        candidates = pool[
            ~pool["Nama Makanan"].astype(str).str.strip().str.lower().isin(blocked)
        ].sort_values("Distance").head(top_k).copy()

        if len(candidates) == 0:
            return

        candidates["prob"] = 1 / (candidates["Distance"] + 1e-9)
        candidates["prob"] = candidates["prob"] / candidates["prob"].sum()

        attempts = 0
        while len(target_list) < max_count and len(candidates) > 0 and attempts < 20:
            attempts += 1
            selected = candidates.sample(1, weights="prob", replace=False)
            if _calc_current_calories() + selected["Kalori (Kkal)"].values[0] <= tdee:
                target_list.append(selected)
                nama = selected["Nama Makanan"].values[0]
                norm_nama = _normalize_name(nama)
                used_set.add(norm_nama)
                if history is not None:
                    history[norm_nama] = current_day
                blocked.add(norm_nama)
                candidates = candidates[~candidates["Nama Makanan"].isin([nama])]
            else:
                candidates = candidates[~candidates["Nama Makanan"].isin(selected["Nama Makanan"].tolist())]

    # ---- MAKAN PAGI ----
    _try_add_from_pool(meal_pagi, selected_meals, used_meals, 1, meal_history, top_k=21)

    # ---- MAKAN SIANG & MALAM ----
    _try_add_from_pool(meal_siang_malam, selected_meals, used_meals, 3, meal_history, top_k=21)

    # ---- CAMILAN (2) ----
    if len(snack_minuman) > 0:
        _try_add_from_pool(snack_minuman, selected_snacks, used_snacks, 1, snack_history, top_k=14)
    if len(selected_snacks) < 1:
        _try_add_from_pool(snack_padat, selected_snacks, used_snacks, 1, snack_history, top_k=14)
    if len(selected_snacks) < 1:
        _try_add_from_pool(snack_data, selected_snacks, used_snacks, 1, snack_history, top_k=14)

    # Pilih camilan kedua (berlawanan jenis dengan camilan pertama)
    if len(selected_snacks) >= 1:
        first_snack_name = selected_snacks[0]["Nama Makanan"].values[0]
        second_pool = snack_padat if is_beverage(first_snack_name) else snack_minuman
        _try_add_from_pool(second_pool, selected_snacks, used_snacks, 2, snack_history, top_k=14)

    if selected_meals or selected_snacks:
        return pd.concat(selected_meals + selected_snacks, ignore_index=True)
    return pd.DataFrame()


# ===== EVALUASI =====
def evaluate_compliance(weekly_summary, batas):
    natrium_limit    = batas["natrium_meal"]    * 3 + batas["natrium_snack"]    * 2
    gula_limit       = batas["gula_meal"]       * 3 + batas["gula_snack"]       * 2
    lemak_limit      = batas["lemak_meal"]      * 3 + batas["lemak_snack"]      * 2
    kolesterol_limit = batas["kolesterol_meal"] * 3 + batas["kolesterol_snack"] * 2

    results = []
    for day in weekly_summary:
        n_ok = day["Total Natrium (mg)"]    <= natrium_limit
        g_ok = day["Total Gula (g)"]        <= gula_limit
        l_ok = day["Total Lemak (g)"]       <= lemak_limit
        k_ok = day["Total Kolesterol (mg)"] <= kolesterol_limit
        results.append({
            "Hari":       day["Hari"],
            "Natrium":    "✔" if n_ok else "✘",
            "Gula":       "✔" if g_ok else "✘",
            "Lemak":      "✔" if l_ok else "✘",
            "Kolesterol": "✔" if k_ok else "✘",
            "Sesuai":     n_ok and g_ok and l_ok and k_ok
        })

    compliance_rate = sum(r["Sesuai"] for r in results) / 7 * 100
    return results, round(compliance_rate, 2)


# ===== FUNGSI UTAMA =====
def run_pipeline(age, gender, weight, height, activity, systolic, diastolic,
                 allergy=None, rice_option="none", dataset_path="food_datasets.csv"):
    np.random.seed(42)
    if systolic < 140 and diastolic < 90:
        return {
            "allowed": False,
            "message": "Maaf, hasil tekanan darah dalam kategori normal atau belum termasuk hipertensi. Sistem ini hanya dapat digunakan untuk pengguna dengan hipertensi (≥ 140/90 mmHg)."
        }

    df         = load_dataset(dataset_path)
    bmr        = hitung_bmr(age, gender, weight, height)
    tdee       = hitung_tdee(bmr, activity)
    derajat    = cek_hipertensi(systolic, diastolic)
    if derajat == 0:
        return {
            "allowed": False,
            "message": "Maaf, hasil tekanan darah dalam kategori normal atau belum termasuk hipertensi. Sistem ini hanya dapat digunakan untuk pengguna dengan hipertensi (≥ 140/90 mmHg)."
        }
    batas      = hitung_batas_nutrisi(tdee, derajat, age, gender, weight)
    akg        = get_akg(age, gender)
    koreksi_bb = batas["koreksi_bb"]

    meal  = filter_makanan(df, batas, "meal",  allergy)
    snack = filter_makanan(df, batas, "snack", allergy)

    filtered_counts = {
        "meal": len(meal),
        "snack": len(snack),
        "total": len(meal) + len(snack)
    }

    if filtered_counts["meal"] == 0 or filtered_counts["snack"] == 0:
        raise ValueError("Tidak ada makanan yang lolos filter. Coba periksa input atau alergi.")

    tabel_nutrisi = build_tabel_nutrisi(batas, tdee, derajat, akg, koreksi_bb)
    tabel_nutrisi["Normalisasi Makan Utama"] = (
        tabel_nutrisi["Target Makan Utama"] / tabel_nutrisi["Referensi Harian (AKG / Batas)"]
    )
    tabel_nutrisi["Normalisasi Camilan"] = (
        tabel_nutrisi["Target Camilan"] / tabel_nutrisi["Referensi Harian (AKG / Batas)"]
    )

    daily_reference = tabel_nutrisi["Referensi Harian (AKG / Batas)"].values
    meal_norm  = akg_normalize(meal,  daily_reference)
    snack_norm = akg_normalize(snack, daily_reference)

    scaler   = MinMaxScaler()
    combined = pd.concat([meal_norm[fitur_cols], snack_norm[fitur_cols]])
    scaler.fit(combined)

    meal_scaled  = meal_norm.copy()
    snack_scaled = snack_norm.copy()
    meal_scaled[fitur_cols]  = scaler.transform(meal_norm[fitur_cols]).clip(0, 1)
    snack_scaled[fitur_cols] = scaler.transform(snack_norm[fitur_cols]).clip(0, 1)

    user_meal_vector  = build_user_vector(tabel_nutrisi, "Normalisasi Makan Utama")
    user_snack_vector = build_user_vector(tabel_nutrisi, "Normalisasi Camilan")

    user_meal_scaled  = scaler.transform(pd.DataFrame([user_meal_vector],  columns=fitur_cols))[0]
    user_snack_scaled = scaler.transform(pd.DataFrame([user_snack_vector], columns=fitur_cols))[0]

    meal_scaled["Distance"]  = weighted_euclidean(user_meal_scaled,  meal_scaled)
    snack_scaled["Distance"] = weighted_euclidean(user_snack_scaled, snack_scaled)

    meal_sorted  = meal_scaled.sort_values("Distance").reset_index(drop=True)
    snack_sorted = snack_scaled.sort_values("Distance").reset_index(drop=True)

    # ===== RANKING WED =====
    meal_distance_all = meal_sorted.drop_duplicates(subset=["Nama Makanan"], keep="first")[["Nama Makanan", "Distance"]]
    knn_rank_meal = meal_distance_all.merge(
        meal[["Nama Makanan", "Kategori", "Natrium (Mg)", "Lemak (G)", "Kolesterol (Mg)", "Gula (G)"]].drop_duplicates(subset=["Nama Makanan"]),
        on="Nama Makanan", how="left"
    ).rename(columns={"Natrium (Mg)": "Natrium", "Lemak (G)": "Lemak", "Kolesterol (Mg)": "Kolesterol", "Gula (G)": "Gula"})
    knn_rank_meal = knn_rank_meal.sort_values("Distance").reset_index(drop=True)
    knn_rank_meal["Rank"] = knn_rank_meal.index + 1
    knn_rank_meal = knn_rank_meal.to_dict(orient="records")

    snack_distance_all = snack_sorted.drop_duplicates(subset=["Nama Makanan"], keep="first")[["Nama Makanan", "Distance"]]
    knn_rank_snack = snack_distance_all.merge(
        snack[["Nama Makanan", "Kategori", "Natrium (Mg)", "Lemak (G)", "Kolesterol (Mg)", "Gula (G)"]].drop_duplicates(subset=["Nama Makanan"]),
        on="Nama Makanan", how="left"
    ).rename(columns={"Natrium (Mg)": "Natrium", "Lemak (G)": "Lemak", "Kolesterol (Mg)": "Kolesterol", "Gula (G)": "Gula"})
    knn_rank_snack = knn_rank_snack.sort_values("Distance").reset_index(drop=True)
    knn_rank_snack["Rank"] = knn_rank_snack.index + 1
    knn_rank_snack = knn_rank_snack.to_dict(orient="records")

    # ===== CLUSTERING (HANYA PROFIL RISIKO) =====
    meal_clustered  = apply_clustering(meal_sorted,  meal,  batas, "meal")
    snack_clustered = apply_clustering(snack_sorted, snack, batas, "snack")

    meal_clustered = meal_clustered.sort_values("Distance").reset_index(drop=True)
    meal_clustered["Rank"] = meal_clustered.index + 1
    snack_clustered = snack_clustered.sort_values("Distance").reset_index(drop=True)
    snack_clustered["Rank"] = snack_clustered.index + 1

    meal_pagi_clustered  = meal_clustered[meal_clustered["Nama Makanan"].apply(is_pagi_food)].copy()
    meal_siang_clustered = meal_clustered[~meal_clustered["Nama Makanan"].apply(is_pagi_food)].copy()

    if len(meal_pagi_clustered) == 0:
        meal_pagi_clustered = meal_clustered.copy()

    rice_data = get_rice_data(df, rice_option)

    meal_history, snack_history = {}, {}
    weekly_summary = []
    weekly_detail  = []

    for day in range(1, 8):
        nama_hari = NAMA_HARI[day - 1]
        day_used_meals = set()
        day_used_snacks = set()

        daily_plan = generate_daily_plan(
            meal_pagi_clustered, meal_siang_clustered, snack_clustered,
            day_used_meals, day_used_snacks, tdee, batas,
            current_day=day, meal_history=meal_history, snack_history=snack_history
        )

        # plan_info sekarang hanya menyimpan Profil Risiko dan Jarak
        plan_info = {
            row["Nama Makanan"]: {
                "Profil Risiko": row["Profil Risiko"],
                "Jarak": round(float(row["Distance"]), 4),
                "Cluster": int(row["Cluster"]) if not pd.isna(row["Cluster"]) else None
            }
            for _, row in daily_plan.iterrows()
        }

        selected_names = daily_plan["Nama Makanan"].tolist()

        daily_real = pd.concat([
            meal[meal["Nama Makanan"].isin(selected_names)],
            snack[snack["Nama Makanan"].isin(selected_names)]
        ]).copy().reset_index(drop=True)

        waktu_map = {}
        meal_idx  = [i for i in range(len(daily_real)) if daily_real.at[i, "Kategori"] == "Meal"]
        snack_idx = [i for i in range(len(daily_real)) if daily_real.at[i, "Kategori"] == "Snack"]

        if len(meal_idx) >= 1:
            waktu_map[meal_idx[0]] = "Makan Pagi"
        if len(meal_idx) >= 2:
            waktu_map[meal_idx[1]] = "Makan Siang"
        if len(meal_idx) >= 3:
            waktu_map[meal_idx[2]] = "Makan Malam"
        if len(snack_idx) >= 1:
            waktu_map[snack_idx[0]] = "Camilan 1"
        if len(snack_idx) >= 2:
            waktu_map[snack_idx[1]] = "Camilan 2"

        daily_real["Waktu Makan"]    = [waktu_map.get(i, "-") for i in range(len(daily_real))]
        daily_real["Nama Tampilan"]  = daily_real["Nama Makanan"].copy()
        daily_real["Karbo Tambahan"] = None
        daily_real["Karbo Bahan"]    = None
        daily_real["Karbo Resep"]    = None

        if rice_data is not None:
            for i in meal_idx[:3]:
                nama_asli = str(daily_real.at[i, "Nama Makanan"])
                if is_rice_compatible(nama_asli):
                    nama_nasi = rice_data["Nama Makanan"]
                    daily_real.at[i, "Karbo Tambahan"] = nama_nasi
                    daily_real.at[i, "Karbo Bahan"]    = rice_data["Bahan"]
                    daily_real.at[i, "Karbo Resep"]    = rice_data["Resep"]
                    for col in NUTRISI_COLS:
                        daily_real.at[i, col] = float(daily_real.at[i, col]) + float(rice_data[col])
                    daily_real.at[i, "Bahan"] = (
                        str(daily_real.at[i, "Bahan"]) + f'; {rice_data["Bahan"]}'
                    )
                    daily_real.at[i, "Resep"] = (
                        str(daily_real.at[i, "Resep"]) + f' | Sajikan bersama: {rice_data["Resep"]}'
                    )

        weekly_summary.append({
            "Hari":                  nama_hari,
            "Total Natrium (mg)":    round(daily_real["Natrium (Mg)"].sum(), 2),
            "Total Gula (g)":        round(daily_real["Gula (G)"].sum(),     2),
            "Total Kalori (Kkal)":   round(daily_real["Kalori (Kkal)"].sum(), 2),
            "Total Lemak (g)":       round(daily_real["Lemak (G)"].sum(),    2),
            "Total Kolesterol (mg)": round(daily_real["Kolesterol (Mg)"].sum(), 2),
        })

        for i, (_, row) in enumerate(daily_real.iterrows()):
            nama_tampilan = str(row["Nama Tampilan"])
            nama_original = str(row["Nama Makanan"])
            info = plan_info.get(nama_original, {"Profil Risiko": "-", "Jarak": 0.0})

            weekly_detail.append({
                "Hari":                  nama_hari,
                "Waktu Makan":           row["Waktu Makan"],
                "Nama Makanan":          nama_tampilan,
                "Kategori":              row["Kategori"],
                "Profil Risiko":         info["Profil Risiko"],
                "Jarak":                 info["Jarak"],
                "Kalori (Kkal)":         round(float(row.get("Kalori (Kkal)", 0)), 2),
                "Protein (G)":           round(float(row.get("Protein (G)", 0)), 2),
                "Lemak (G)":             round(float(row.get("Lemak (G)", 0)), 2),
                "Karbohidrat (G)":       round(float(row.get("Karbohidrat (G)", 0)), 2),
                "Serat (G)":             round(float(row.get("Serat (G)", 0)), 2),
                "Kalium (Mg)":           round(float(row.get("Kalium (Mg)", 0)), 2),
                "Natrium (Mg)":          round(float(row.get("Natrium (Mg)", 0)), 2),
                "Gula (G)":              round(float(row.get("Gula (G)", 0)), 2),
                "Kolesterol (Mg)":       round(float(row.get("Kolesterol (Mg)", 0)), 2),
                "Bahan":                 row.get("Bahan", "-"),
                "Resep":                 row.get("Resep", "-"),
                "Karbo Tambahan":        row.get("Karbo Tambahan", None),
                "Karbo Bahan":           row.get("Karbo Bahan", None),
                "Karbo Resep":           row.get("Karbo Resep", None),
                "Karbo Kalori (Kkal)":   round(float(rice_data["Kalori (Kkal)"]), 2)
                    if row.get("Karbo Tambahan") and rice_data is not None else None,
                "Karbo Protein (G)":     round(float(rice_data["Protein (G)"]), 2)
                    if row.get("Karbo Tambahan") and rice_data is not None else None,
                "Karbo Lemak (G)":       round(float(rice_data["Lemak (G)"]), 2)
                    if row.get("Karbo Tambahan") and rice_data is not None else None,
                "Karbo Karbohidrat (G)": round(float(rice_data["Karbohidrat (G)"]), 2)
                    if row.get("Karbo Tambahan") and rice_data is not None else None,
                "Karbo Serat (G)":       round(float(rice_data["Serat (G)"]), 2)
                    if row.get("Karbo Tambahan") and rice_data is not None else None,
                "Karbo Kalium (Mg)":     round(float(rice_data["Kalium (Mg)"]), 2)
                    if row.get("Karbo Tambahan") and rice_data is not None else None,
                "Karbo Natrium (Mg)":    round(float(rice_data["Natrium (Mg)"]), 2)
                    if row.get("Karbo Tambahan") and rice_data is not None else None,
                "Karbo Gula (G)":        round(float(rice_data["Gula (G)"]), 2)
                    if row.get("Karbo Tambahan") and rice_data is not None else None,
                "Karbo Kolesterol (Mg)": round(float(rice_data["Kolesterol (Mg)"]), 2)
                    if row.get("Karbo Tambahan") and rice_data is not None else None,
            })

    compliance_results, compliance_rate = evaluate_compliance(weekly_summary, batas)
    total_menus     = len(weekly_detail)
    unique_menus    = len(set(d["Nama Makanan"] for d in weekly_detail))
    diversity_score = round(unique_menus / total_menus * 100, 2)
    avg_distance    = round(sum(d["Jarak"] for d in weekly_detail) / total_menus, 4)

    return {
        "bmr":             round(bmr, 2),
        "tdee":            round(tdee, 2),
        "derajat":         derajat,
        "batas":           batas,
        "filtered_counts": filtered_counts,
        "tabel_nutrisi":   tabel_nutrisi.round(4).to_dict(orient="records"),
        "weekly_summary":  weekly_summary,
        "weekly_detail":   weekly_detail,
        "knn_rank_meal":   knn_rank_meal,
        "knn_rank_snack":  knn_rank_snack,
        "compliance":      compliance_results,
        "compliance_rate": compliance_rate,
        "diversity_score": diversity_score,
        "avg_distance":    avg_distance,
        "meal_clustered":  meal_clustered.to_dict(orient="records"),
        "snack_clustered": snack_clustered.to_dict(orient="records"),
    }