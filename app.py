# Nama File: app.py

import os
import uuid
import numpy as np
import matplotlib
matplotlib.use('Agg') # Mode non-interaktif untuk server
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask import (
    Flask, request, render_template, flash, 
    redirect, url_for, session, send_from_directory
)
from werkzeug.utils import secure_filename

# Pastikan file topoindex_web.py ada di direktori yang sama
# dan berisi fungsi run_topoindex_analysis() dan read_ascii_grid()
from topoindex_web import run_topoindex_analysis, read_ascii_grid

# Inisialisasi aplikasi Flask
app = Flask(__name__)
# Kunci rahasia sangat penting untuk keamanan sesi
app.secret_key = 'ganti-dengan-kunci-rahasia-anda-yang-sangat-aman'

# --- KONFIGURASI APLIKASI ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'outputs')

# Buat folder jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Ekstensi file yang diizinkan untuk diunggah
ALLOWED_EXTENSIONS = {'asc', 'txt'}

# --- FUNGSI BANTU ---

def allowed_file(filename):
    """Memeriksa apakah ekstensi file diizinkan."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_hillshade_image(asc_path):
    """Membuat gambar hillshade dari file grid ASCII dan mengembalikannya sebagai string Base64."""
    try:
        elev_grid, header = read_ascii_grid(asc_path)
        nodata_value = int(header.get('nodata_value', -9999))
        elev_grid = np.ma.masked_where(elev_grid == nodata_value, elev_grid)

        # Parameter standar untuk hillshade
        azimuth, altitude = 315, 45
        azimuth_rad, altitude_rad = np.radians(azimuth), np.radians(altitude)
        
        # Kalkulasi Hillshade menggunakan NumPy
        x, y = np.gradient(elev_grid)
        slope = np.pi / 2. - np.arctan(np.sqrt(x*x + y*y))
        aspect = np.arctan2(-x, y)
        shaded = np.sin(altitude_rad) * np.sin(slope) + \
                 np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
        shaded = np.clip(shaded, 0, 1)

        # Membuat gambar dengan Matplotlib
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(shaded, cmap='gray', vmin=0, vmax=1)
        ax.set_axis_off()
        fig.tight_layout(pad=0)

        # Menyimpan gambar ke buffer di memori
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        # Meng-encode gambar menjadi Base64
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error saat membuat hillshade: {e}")
        return None

# --- ROUTE (URL) APLIKASI ---

@app.route('/')
def index():
    """Menampilkan halaman utama/beranda dengan deskripsi aplikasi."""
    return render_template('index.html') 

@app.route('/analysis')
def analysis_page():
    """Menampilkan halaman dengan formulir untuk memulai analisis."""
    return render_template('analysis.html')

@app.route('/contact')
def contact():
    """Menampilkan halaman kontak."""
    return render_template('contact.html')

@app.route('/process', methods=['POST'])
def process():
    """Menangani unggahan file, menjalankan analisis, dan mengarahkan ke halaman hasil."""
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    if 'dem_file' not in request.files or 'dir_file' not in request.files or \
       request.files['dem_file'].filename == '' or request.files['dir_file'].filename == '':
        flash('Kesalahan: Kedua file (DEM dan Arah Aliran) wajib diunggah.')
        return redirect(url_for('analysis_page'))

    dem_file = request.files['dem_file']
    dir_file = request.files['dir_file']

    if allowed_file(dem_file.filename) and allowed_file(dir_file.filename):
        dem_filename = secure_filename(dem_file.filename)
        session['dem_filename'] = dem_filename  # Simpan nama file DEM untuk visualisasi
        
        dem_path = os.path.join(app.config['UPLOAD_FOLDER'], dem_filename)
        dir_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dir_file.filename))
        dem_file.save(dem_path)
        dir_file.save(dir_path)

        try:
            pwr = float(request.form['exponent'])
            itmax = int(request.form['iterations'])
            suffix = request.form['suffix']
            
            run_topoindex_analysis(
                dem_path=dem_path, dir_path=dir_path, output_folder=output_dir, 
                suffix=suffix, pwr=pwr, itmax=itmax
            )
            return redirect(url_for('results'))

        except Exception as e:
            flash(f"Terjadi error saat analisis: {e}")
            return redirect(url_for('analysis_page'))
    else:
        flash("Ekstensi file tidak diizinkan. Gunakan .asc atau .txt")
        return redirect(url_for('analysis_page'))

@app.route('/results')
def results():
    """Menampilkan halaman hasil dengan daftar file dan visualisasi."""
    session_id = session.get('session_id')
    
    if not session_id:
        flash("Tidak ada sesi analisis yang aktif. Silakan mulai analisis baru.")
        return redirect(url_for('analysis_page'))
    
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    
    if not os.path.exists(output_dir):
        flash("Folder hasil untuk sesi ini tidak ditemukan. Silakan mulai analisis baru.")
        return redirect(url_for('analysis_page'))
        
    output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    
    hillshade_img = None
    file_to_visualize = request.args.get('visualize')

    if file_to_visualize:
        # --- LOGIKA BARU UNTUK MEMILIH FILE ---
        file_path = None
        if file_to_visualize == 'dem':
            # Jika diminta untuk visualisasi DEM asli
            dem_filename = session.get('dem_filename')
            if dem_filename:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], dem_filename)
            else:
                flash("Nama file DEM tidak ditemukan di sesi ini.")
        elif file_to_visualize in output_files:
            # Jika nama file yang diminta ada di dalam daftar output_files
            file_path = os.path.join(output_dir, file_to_visualize)
        else:
            flash(f"File '{file_to_visualize}' tidak ditemukan untuk divisualisasikan.")

        # Jika path file valid, buat gambar hillshade
        if file_path and os.path.exists(file_path):
            hillshade_img = generate_hillshade_image(file_path)
        elif file_path:
            flash(f"Path file '{file_path}' tidak ditemukan di server.")
            
    return render_template('results.html', 
                           output_files=output_files, 
                           session_id=session_id,
                           hillshade_img=hillshade_img)

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    """Menyediakan fungsionalitas unduh untuk file hasil."""
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    return send_from_directory(directory=output_dir, path=filename, as_attachment=True)

# --- MENJALANKAN APLIKASI ---
if __name__ == '__main__':
    # host='0.0.0.0' membuat server dapat diakses dari luar jaringan lokal
    app.run(host='0.0.0.0', port=5000, debug=True)
