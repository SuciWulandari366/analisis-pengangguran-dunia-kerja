from airflow.decorators import dag, task
from datetime import datetime
import requests
import pandas as pd
from airflow.providers.mysql.hooks.mysql import MySqlHook


# =========================================================
# KONFIGURASI
# =========================================================

API_KEY = "08e5b8e24a3ec4f37a9d278090518c34"

API_URL = "https://webapi.bps.go.id/v1/api/list"

MYSQL_CONN_ID = "mysql_data_science"


# =========================================================
# KODE TAHUN BPS
# =========================================================

TAHUN_BPS = {
    "2021": "121",
    "2022": "122",
    "2023": "123",
    "2024": "124",
    "2025": "125"
}


# =========================================================
# DAG
# =========================================================

@dag(
    dag_id="etl_pengangguran",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["data-science-2", "bps"],
)
def etl_pengangguran():

    # =====================================================
    # =====================================================
    # TABEL 1
    # VAR 2479
    # Penduduk 15 Tahun Ke Atas yang Bekerja
    # menurut Lapangan Usaha
    # =====================================================
    # =====================================================

    @task
    def extract_pekerja():

        semua_data = []

        for tahun_label, tahun_id in TAHUN_BPS.items():

            print("")
            print("=" * 60)
            print("EXTRACT TABEL 1")
            print("Tahun:", tahun_label)
            print("Kode tahun BPS:", tahun_id)
            print("=" * 60)

            params = {
                "model": "data",
                "lang": "ind",
                "domain": "0000",
                "var": "2479",
                "th": tahun_id,
                "key": API_KEY
            }

            response = requests.get(
                API_URL,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            if data.get("status") != "OK":
                raise Exception(
                    f"API BPS gagal untuk tahun "
                    f"{tahun_label}: {data}"
                )

            if "datacontent" not in data:
                raise Exception(
                    f"Tabel 1 tidak memiliki datacontent "
                    f"untuk tahun {tahun_label} "
                    f"(kode {tahun_id}). Response: {data}"
                )

            print(
                "Variabel:",
                data["var"][0]["label"]
            )

            print(
                "Tahun API:",
                data["tahun"][0]["label"]
            )

            print(
                "Jumlah data:",
                len(data["datacontent"])
            )

            semua_data.append(data)

        return semua_data


    @task
    def transform_pekerja(data_list):

        rows = []

        for data in data_list:

            # Mapping lapangan usaha
            sektor = {
                str(item["val"]): item["label"]
                for item in data["vervar"]
            }

            # Mapping periode
            periode = {
                str(item["val"]): item["label"].strip()
                for item in data["turtahun"]
            }

            var_id = str(
                data["var"][0]["val"]
            )

            tahun_id = str(
                data["tahun"][0]["val"]
            )

            tahun_label = data["tahun"][0]["label"]

            # Tabel 1 menggunakan tahun 4 digit
            tahun_key = tahun_id.zfill(4)

            print("")
            print("=" * 60)
            print("TRANSFORM TABEL 1")
            print("Tahun:", tahun_label)
            print("Variable ID:", var_id)
            print("Tahun ID:", tahun_id)
            print("Tahun key:", tahun_key)
            print("=" * 60)

            for key, value in data["datacontent"].items():

                key = str(key)

                for periode_id, periode_label in periode.items():

                    suffix = (
                        var_id
                        + tahun_key
                        + periode_id
                    )

                    if key.endswith(suffix):

                        sektor_id = key[
                            :-len(suffix)
                        ]

                        if sektor_id in sektor:

                            rows.append({
                                "tahun": int(tahun_label),
                                "periode": periode_label,
                                "lapangan_usaha": sektor[
                                    sektor_id
                                ],
                                "jumlah_pekerja": value
                            })

                        break

        if not rows:

            raise Exception(
                "Tidak ada data Tabel 1 "
                "yang berhasil ditransform."
            )

        df = pd.DataFrame(rows)

        df = df.drop_duplicates()

        df = df.sort_values(
            [
                "tahun",
                "periode",
                "lapangan_usaha"
            ]
        ).reset_index(drop=True)

        print("")
        print("===== HASIL TABEL 1 =====")
        print(df.to_string(index=False))
        print("Jumlah baris:", len(df))

        return df.to_dict(
            orient="records"
        )


    @task
    def load_pekerja(rows):

        hook = MySqlHook(
            mysql_conn_id=MYSQL_CONN_ID
        )

        records = [
            (
                row["tahun"],
                row["periode"],
                row["lapangan_usaha"],
                row["jumlah_pekerja"]
            )
            for row in rows
        ]

        hook.insert_rows(
            table="produksi_tenaga_kerja",
            rows=records,
            target_fields=[
                "tahun",
                "periode",
                "lapangan_usaha",
                "jumlah_pekerja"
            ],
            commit_every=100
        )

        print("")
        print("===== LOAD TABEL 1 =====")
        print("Jumlah baris:", len(records))


    # =====================================================
    # =====================================================
    # TABEL 2
    # VAR 543
    # Tingkat Pengangguran Terbuka Menurut Provinsi
    # =====================================================
    # =====================================================

    @task
    def extract_pengangguran():

        semua_data = []

        for tahun_label, tahun_id in TAHUN_BPS.items():

            print("")
            print("=" * 60)
            print("EXTRACT TABEL 2")
            print("Tahun:", tahun_label)
            print("Kode tahun BPS:", tahun_id)
            print("=" * 60)

            params = {
                "model": "data",
                "lang": "ind",
                "domain": "0000",
                "var": "543",
                "th": tahun_id,
                "key": API_KEY
            }

            response = requests.get(
                API_URL,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            if data.get("status") != "OK":
                raise Exception(
                    f"API BPS gagal untuk tahun "
                    f"{tahun_label}: {data}"
                )

            if "datacontent" not in data:
                raise Exception(
                    f"Tabel 2 tidak memiliki datacontent "
                    f"untuk tahun {tahun_label} "
                    f"(kode {tahun_id}). Response: {data}"
                )

            print(
                "Variabel:",
                data["var"][0]["label"]
            )

            print(
                "Tahun API:",
                data["tahun"][0]["label"]
            )

            print(
                "Jumlah data:",
                len(data["datacontent"])
            )

            semua_data.append(data)

        return semua_data


    @task
    def transform_pengangguran(data_list):

        rows = []

        for data in data_list:

            # Mapping wilayah
            wilayah = {
                str(item["val"]): item["label"]
                for item in data["vervar"]
            }

            # Mapping periode
            periode = {
                str(item["val"]): item["label"].strip()
                for item in data["turtahun"]
            }

            var_id = str(
                data["var"][0]["val"]
            )

            tahun_id = str(
                data["tahun"][0]["val"]
            )

            tahun_label = data["tahun"][0]["label"]

            # Tabel 2 juga menggunakan tahun 4 digit
            tahun_key = tahun_id.zfill(4)

            print("")
            print("=" * 60)
            print("TRANSFORM TABEL 2")
            print("Tahun:", tahun_label)
            print("Variable ID:", var_id)
            print("Tahun ID:", tahun_id)
            print("Tahun key:", tahun_key)
            print("=" * 60)

            for key, value in data["datacontent"].items():

                key = str(key)

                for periode_id, periode_label in periode.items():

                    suffix = (
                        var_id
                        + tahun_key
                        + periode_id
                    )

                    if key.endswith(suffix):

                        wilayah_id = key[
                            :-len(suffix)
                        ]

                        if wilayah_id in wilayah:

                            rows.append({
                                "tahun": int(tahun_label),
                                "periode": periode_label,
                                "wilayah": wilayah[
                                    wilayah_id
                                ],
                                "tingkat_pengangguran": value
                            })

                        break

        if not rows:

            raise Exception(
                "Tidak ada data Tabel 2 "
                "yang berhasil ditransform."
            )

        df = pd.DataFrame(rows)

        df = df.drop_duplicates()

        df = df.sort_values(
            [
                "tahun",
                "periode",
                "wilayah"
            ]
        ).reset_index(drop=True)

        print("")
        print("===== HASIL TABEL 2 =====")
        print(df.to_string(index=False))
        print("Jumlah baris:", len(df))

        return df.to_dict(
            orient="records"
        )


    @task
    def load_pengangguran(rows):

        hook = MySqlHook(
            mysql_conn_id=MYSQL_CONN_ID
        )

        records = [
            (
                row["tahun"],
                row["periode"],
                row["wilayah"],
                row["tingkat_pengangguran"]
            )
            for row in rows
        ]

        hook.insert_rows(
            table="tingkat_pengangguran",
            rows=records,
            target_fields=[
                "tahun",
                "periode",
                "wilayah",
                "tingkat_pengangguran"
            ],
            commit_every=100
        )

        print("")
        print("===== LOAD TABEL 2 =====")
        print("Jumlah baris:", len(records))


    # =====================================================
    # =====================================================
    # TABEL 3
    # VAR 2333
    # Penduduk Bekerja Menurut Provinsi
    # dan Lapangan Usaha
    # =====================================================
    # =====================================================

    @task
    def extract_pekerja_provinsi():

        semua_data = []

        for tahun_label, tahun_id in TAHUN_BPS.items():

            print("")
            print("=" * 60)
            print("EXTRACT TABEL 3")
            print("Tahun:", tahun_label)
            print("Kode tahun BPS:", tahun_id)
            print("=" * 60)

            params = {
                "model": "data",
                "lang": "ind",
                "domain": "0000",
                "var": "2333",
                "th": tahun_id,
                "key": API_KEY
            }

            response = requests.get(
                API_URL,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            if data.get("status") != "OK":
                raise Exception(
                    f"API BPS gagal untuk tahun "
                    f"{tahun_label}: {data}"
                )

            if "datacontent" not in data:
                raise Exception(
                    f"Tabel 3 tidak memiliki datacontent "
                    f"untuk tahun {tahun_label} "
                    f"(kode {tahun_id}). Response: {data}"
                )

            print(
                "Variabel:",
                data["var"][0]["label"]
            )

            print(
                "Tahun API:",
                data["tahun"][0]["label"]
            )

            print(
                "Jumlah data:",
                len(data["datacontent"])
            )

            semua_data.append(data)

        return semua_data


    @task
    def transform_pekerja_provinsi(data_list):

        rows = []

        for data in data_list:

            # Mapping wilayah
            wilayah = {
                str(item["val"]): item["label"]
                for item in data["vervar"]
            }

            # Mapping sektor
            sektor = {
                str(item["val"]): item["label"]
                for item in data["turvar"]
            }

            # Mapping periode
            periode = {
                str(item["val"]): item["label"].strip()
                for item in data["turtahun"]
            }

            var_id = str(
                data["var"][0]["val"]
            )

            tahun_id = str(
                data["tahun"][0]["val"]
            )

            tahun_label = data["tahun"][0]["label"]

            print("")
            print("=" * 60)
            print("TRANSFORM TABEL 3")
            print("Tahun:", tahun_label)
            print("Variable ID:", var_id)
            print("Tahun ID:", tahun_id)
            print("=" * 60)

            for key, value in data["datacontent"].items():

                key = str(key)

                # Tabel 3:
                #
                # wilayah + var + sektor + tahun + periode
                #
                # 4 + 4 + 4 + 3 + 3 = 18 digit

                if len(key) != 18:
                    continue

                wilayah_id = key[0:4]
                var_key = key[4:8]
                sektor_id = key[8:12]
                tahun_key = key[12:15]
                periode_id = key[15:18]

                if var_key != var_id:
                    continue

                if tahun_key != tahun_id:
                    continue

                if wilayah_id not in wilayah:
                    continue

                if sektor_id not in sektor:
                    continue

                if periode_id not in periode:
                    continue

                rows.append({
                    "tahun": int(tahun_label),
                    "periode": periode[
                        periode_id
                    ],
                    "wilayah": wilayah[
                        wilayah_id
                    ],
                    "lapangan_usaha": sektor[
                        sektor_id
                    ],
                    "jumlah_pekerja": value
                })

        if not rows:

            raise Exception(
                "Tidak ada data Tabel 3 "
                "yang berhasil ditransform."
            )

        df = pd.DataFrame(rows)

        df = df.drop_duplicates()

        df = df.sort_values(
            [
                "tahun",
                "periode",
                "wilayah",
                "lapangan_usaha"
            ]
        ).reset_index(drop=True)

        print("")
        print("===== HASIL TABEL 3 =====")
        print(
            df.head(30).to_string(index=False)
        )

        print(
            "Jumlah baris:",
            len(df)
        )

        return df.to_dict(
            orient="records"
        )


    @task
    def load_pekerja_provinsi(rows):

        hook = MySqlHook(
            mysql_conn_id=MYSQL_CONN_ID
        )

        records = [
            (
                row["tahun"],
                row["periode"],
                row["wilayah"],
                row["lapangan_usaha"],
                row["jumlah_pekerja"]
            )
            for row in rows
        ]

        hook.insert_rows(
            table="pekerja_provinsi",
            rows=records,
            target_fields=[
                "tahun",
                "periode",
                "wilayah",
                "lapangan_usaha",
                "jumlah_pekerja"
            ],
            commit_every=100
        )

        print("")
        print("===== LOAD TABEL 3 =====")
        print("Jumlah baris:", len(records))


    # =====================================================
    # DEPENDENCY TABEL 1
    # =====================================================

    data_pekerja = extract_pekerja()

    hasil_pekerja = transform_pekerja(
        data_pekerja
    )

    load_pekerja(
        hasil_pekerja
    )


    # =====================================================
    # DEPENDENCY TABEL 2
    # =====================================================

    data_pengangguran = extract_pengangguran()

    hasil_pengangguran = transform_pengangguran(
        data_pengangguran
    )

    load_pengangguran(
        hasil_pengangguran
    )


    # =====================================================
    # DEPENDENCY TABEL 3
    # =====================================================

    data_pekerja_provinsi = extract_pekerja_provinsi()

    hasil_pekerja_provinsi = transform_pekerja_provinsi(
        data_pekerja_provinsi
    )

    load_pekerja_provinsi(
        hasil_pekerja_provinsi
    )


# =========================================================
# AKTIFKAN DAG
# =========================================================

etl_pengangguran()