# ===== app.py =====
from flask import Flask, render_template, request, make_response
import traceback
import gc
import pipeline as pl

app = Flask(__name__)

# =========================
# 1. CEKAL CACHE BROWSER (SOLUSI UTAMA)
# =========================
@app.after_request
def add_header(response):
    """
    Mencegah browser menyimpan cache dari POST request.
    Ini memaksa browser untuk selalu meminta data baru ke server setiap kali user submit form.
    """
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # Paksa Python membersihkan memory yang tidak terpakai dari request sebelumnya
        gc.collect()

        # =========================
        # Ambil & Validasi Input Form
        # =========================
        age = int(request.form["age"])
        gender = request.form["gender"].strip().lower()
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        activity = int(request.form["activity"])
        systolic = int(request.form["systolic"])
        diastolic = int(request.form["diastolic"])

        # =========================
        # Pilihan Karbo Tambahan
        # =========================
        rice_option = request.form.get("rice_option", "none").strip().lower()
        valid_rice_options = ["none", "nasi_putih", "nasi_merah", "nasi_kuning", "nasi_uduk"]
        if rice_option not in valid_rice_options:
            rice_option = "none"

        # =========================
        # Alergi
        # =========================
        allergy_raw = request.form.get("allergy", "").strip()
        allergy = allergy_raw if allergy_raw else None

        # =========================
        # Jalankan Pipeline
        # =========================
        result = pl.run_pipeline(
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            activity=activity,
            systolic=systolic,
            diastolic=diastolic,
            allergy=allergy,
            rice_option=rice_option
        )

        # Cek apakah pipeline mengembalikan error (misal: tensi normal)
        if isinstance(result, dict) and not result.get("allowed", True):
            return render_template("index.html", error=result.get("message", "Input tidak memenuhi kriteria."))

        # =========================
        # Format User Input untuk Tampilan
        # =========================
        rice_display_map = {
            "none": "Tidak ada",
            "nasi_putih": "Nasi Putih",
            "nasi_merah": "Nasi Merah",
            "nasi_kuning": "Nasi Kuning",
            "nasi_uduk": "Nasi Uduk",
        }

        user_input = {
            "age": age,
            "gender": gender.title(),
            "weight": weight,
            "height": height,
            "activity": activity,
            "systolic": systolic,
            "diastolic": diastolic,
            "allergy": allergy or "Tidak ada",
            "rice_option": rice_display_map.get(rice_option, "Tidak ada"),
        }

        # =========================
        # Mapping Label Karbo Tambahan di Weekly Detail
        # =========================
        rice_label_map = {
            "nasi putih": "Nasi Putih",
            "nasi merah": "Nasi Merah",
            "nasi kuning": "Nasi Kuning",
            "nasi uduk": "Nasi Uduk",
        }

        for item in result.get("weekly_detail", []):
            karbo_val = item.get("Karbo Tambahan")
            if karbo_val and str(karbo_val).strip().lower() != "none":
                key_normalized = str(karbo_val).strip().lower()
                item["Karbo Tambahan"] = rice_label_map.get(key_normalized, karbo_val)
            else:
                item["Karbo Tambahan"] = None

        # =========================
        # Render Hasil
        # =========================
        return render_template("result.html", result=result, user_input=user_input)

    except ValueError as e:
        return render_template("index.html", error=f"Input tidak valid: {str(e)}")

    except Exception as e:
        # Print full traceback ke terminal untuk debugging
        print(f"[ERROR] {traceback.format_exc()}")
        return render_template(
            "index.html",
            error=f"Terjadi kesalahan sistem: {str(e)}. Silakan cek log server untuk detail."
        )


if __name__ == "__main__":
    # Gunakan threaded=False untuk memastikan request diproses satu per satu (mencegah race condition)
    app.run(debug=True, threaded=False)