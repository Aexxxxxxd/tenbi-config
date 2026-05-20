import cv2
import numpy as np
import time
import threading
import os
import shutil
import tempfile
import wave
import struct
import math
from collections import Counter
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont


# Проверка предустановленного dlib

try:
    import dlib as _dlib
    _DLIB_IMPORTABLE = True
except ImportError:
    _dlib = None
    _DLIB_IMPORTABLE = False
# Всегда ищите файл .dat рядом с самим focus_monitor.py,
# независимо от того, из какого каталога пользователь запускает скрипт.
_DAT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'shape_predictor_68_face_landmarks.dat')


def _get_ascii_safe_path(src: str) -> str:
    """dlib on Windows cannot open files whose path contains non-ASCII characters
    (e.g. Cyrillic usernames or folder names).  If the path is not pure ASCII
    we copy the .dat file to C:\\Windows\\Temp (always ASCII-safe) and return
    that path instead.  The copy is skipped when the cached file is already up
    to date (same size as the source).
    """
    try:
        src.encode('ascii')
        return src          
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Выберите временный каталог, безопасный для ASCII (C:\Windows\Temp в Windows гарантирован формат ASCII)
    if os.name == 'nt':
        safe_dir = os.path.join(os.environ.get('SYSTEMROOT', 'C:\\Windows'), 'Temp')
    else:
        safe_dir = tempfile.gettempdir()

    dst = os.path.join(safe_dir, 'shape_predictor_68_face_landmarks.dat')

    src_size = os.path.getsize(src)
    dst_size = os.path.getsize(dst) if os.path.exists(dst) else -1

    if src_size != dst_size:
        shutil.copy2(src, dst)

    return dst


# надписи для каждой причины отвлечения внимания


REASON_LABELS: dict[str, str] = {
    'focused':      'Фокус',
    'looking_down': 'Голова опущена вниз',
    'looking_up':   'Голова поднята вверх',
    'turned_left':  'Голова повёрнута влево',
    'turned_right': 'Голова повёрнута вправо',
    'no_face':      'Вы вышли из кадра',
}

REASON_ALERT: dict[str, str] = {
    'looking_down': '⚠  Голова опущена вниз',
    'looking_up':   '⚠  Голова поднята вверх',
    'turned_left':  '⚠  Голова повёрнута влево',
    'turned_right': '⚠  Голова повёрнута вправо',
    'no_face':      '⚠  Вы вышли из кадра',
}



# генерация звука без доп библиотек


def _generate_wav(filename: str, frequencies: list, duration: float = 0.3, volume: float = 0.5, sample_rate: int = 44100):
    """Generate a simple WAV tone file from a list of (freq, dur) tuples."""
    frames = []
    for freq, dur in frequencies:
        n_samples = int(sample_rate * dur)
        for i in range(n_samples):
            t = i / sample_rate
            val = math.sin(2 * math.pi * freq * t) * math.exp(-3 * t / dur)
            packed = struct.pack('<h', int(val * 32767 * volume))
            frames.append(packed)
    data = b''.join(frames)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)


SOUNDS_DIR = os.path.join(os.path.dirname(__file__), 'focus_sounds')

SOUND_DEFINITIONS = {
    'Мягкий сигнал': [
        (880, 0.15), (660, 0.15), (440, 0.25)
    ],
    'Двойной бип': [
        (900, 0.12), (0, 0.06), (900, 0.12)
    ],
    'Колокольчик': [
        (1047, 0.1), (1319, 0.1), (1568, 0.2), (1319, 0.15)
    ],
    'Низкий сигнал': [
        (330, 0.2), (220, 0.3)
    ],
    'Быстрый сигнал': [
        (1000, 0.08), (0, 0.04), (1000, 0.08), (0, 0.04), (1000, 0.08)
    ],
}


def _ensure_sounds():
    os.makedirs(SOUNDS_DIR, exist_ok=True)
    for name, freqs in SOUND_DEFINITIONS.items():
        path = os.path.join(SOUNDS_DIR, name + '.wav')
        if not os.path.exists(path):
            safe_freqs = [(f, d) for f, d in freqs if f > 0]
            if safe_freqs:
                _generate_wav(path, safe_freqs)


def _play_wav(path: str):
    """Play a WAV file cross-platform without extra dependencies."""
    try:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        return
    except ImportError:
        pass
    try:
        import subprocess
        if os.uname().sysname == 'Darwin':
            subprocess.Popen(['afplay', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            for player in ('paplay', 'aplay', 'ffplay'):
                try:
                    subprocess.Popen([player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except FileNotFoundError:
                    continue
    except Exception:
        pass



# статистика


class Statistics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.session_start = time.time()
        self.distraction_count = 0
        self.total_distraction_seconds = 0.0
        self._distraction_start: float | None = None

    def on_distraction_start(self):
        if self._distraction_start is None:
            self._distraction_start = time.time()
            self.distraction_count += 1

    def on_distraction_end(self):
        if self._distraction_start is not None:
            self.total_distraction_seconds += time.time() - self._distraction_start
            self._distraction_start = None

    def current_distraction_seconds(self) -> float:
        if self._distraction_start is not None:
            return self.total_distraction_seconds + (time.time() - self._distraction_start)
        return self.total_distraction_seconds

    def session_seconds(self) -> float:
        return time.time() - self.session_start

    def work_seconds(self) -> float:
        return max(0.0, self.session_seconds() - self.current_distraction_seconds())

    def focus_percent(self) -> float:
        s = self.session_seconds()
        if s < 1:
            return 100.0
        return 100.0 * self.work_seconds() / s

    def fmt(self, secs: float) -> str:
        m, s = divmod(int(secs), 60)
        h, m = divmod(m, 60)
        if h:
            return f'{h}ч {m:02d}м {s:02d}с'
        return f'{m:02d}м {s:02d}с'



# Face detector


# Пределы отклонения, используемые для определения относительной калибровки
_YAW_MARGIN        = 0.40   # ± отклонение носовой части от базовой линии
_PITCH_DOWN_MARGIN = 0.65   # отклонение высоты тона от базовой линии
_PITCH_UP_MARGIN   = 0.58   # отклонение высоты тона от базовой линии
_RATIO_Y_MARGIN    = 0.19   # Отклонение центра лица по шкале Хаара от базовой линии

# Требуется такое количество последовательных кадров без лица, прежде чем появится сообщение "no_face".
# При ~30 кадрах в секунду это составляет ~1,2 с — предотвращает ложные предупреждения при кратковременных отключениях.
_NO_FACE_MIN_FRAMES = 36


class FaceDetector:
    """Detects head pose (focused / looking down|up / turned left|right / no face).

    Calibration is camera-position-agnostic: after calling start_calibration()
    the user looks at the screen normally for CALIB_SECS seconds; the recorded
    baseline pitch/yaw (dlib) or face-centre ratio (Haar) is then used as the
    reference for all subsequent detections.
    """

    CALIB_SECS = 4

    def __init__(self, settings=None):
        # настройки - это необязательный параметр настроек; если он задан, пороговые значения обнаружения
        # считываются непосредственно из него, так что изменения ползунка вступают в силу немедленно.
        self._settings = settings

        # --- настройка dlib (отдельная проверка импорта из файла .dat) ----------
        self.use_dlib   = False
        self.dlib_hint  = None   # None → dlib в порядке; str → понятная человеку причина для резервного копирования Haar

        if _DLIB_IMPORTABLE:
            if os.path.exists(_DAT_FILE):
                size_mb = os.path.getsize(_DAT_FILE) / 1024 / 1024
                if size_mb < 50:
                    # Файл существует, но подозрительно мал — все еще внутри архива
                    self.dlib_hint = (
                        f'Файл найден, но похоже он ещё внутри архива .bz2 (размер {size_mb:.1f} МБ, должно быть ~99 МБ).\n'
                        f'Распакованный файл весит около 99 МБ.\n\n'
                        f'Как правильно распаковать:\n'
                        f'  Windows: правой кнопкой по .bz2 → 7-Zip → «Извлечь здесь»\n'
                        f'  macOS:   двойной клик по .bz2 в Finder\n'
                        f'  Linux:   bunzip2 shape_predictor_68_face_landmarks.dat.bz2\n\n'
                        f'Положите .dat файл в папку:\n{os.path.dirname(_DAT_FILE)}'
                    )
                else:
                    try:
                        safe_path = _get_ascii_safe_path(_DAT_FILE)
                        self.detector  = _dlib.get_frontal_face_detector()
                        self.predictor = _dlib.shape_predictor(safe_path)
                        self.use_dlib  = True
                        self.method_label = 'dlib'
                    except Exception as exc:
                        self.dlib_hint = (
                            f'dlib установлен и файл найден ({size_mb:.0f} МБ), но возникла ошибка:\n{exc}\n\n'
                            f'Попробуйте скачать файл заново:\n'
                            f'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
                        )
            else:
                self.dlib_hint = (
                    f'dlib установлен, но файл не найден.\n\n'
                    f'Скачайте:\nhttp://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n\n'
                    f'Распакуйте архив (.bz2) и положите файл .dat в папку:\n'
                    f'{os.path.dirname(_DAT_FILE)}'
                )
        else:
            self.dlib_hint = None  # dlib не установлен — Haar по умолчанию, без предупреждения

        if not self.use_dlib:
            self._init_haar()

        # состояние калибровки 
        self._baseline_pitch:   float | None = None
        self._baseline_yaw:     float | None = None
        self._baseline_ratio_y: float | None = None
        self._calib_samples: list[dict] = []
        self.calibrated = False

        # Rolling stability window 
        self._window: list[str] = []
        self._window_size = 14   # большее окно = более плавно

        #  Отдельный отказ от no_face (позволяет избежать ложных предупреждений о "левом фрейме")
        self._no_face_streak = 0          # последовательные необработанные кадры no_face
        self._last_face_reason = 'focused'  # последняя , по которой присутствовало лицо

    def _init_haar(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.method_label = 'Haar'

   
    # API для калибровки
   

    def start_calibration(self):
        """Reset and begin collecting calibration samples."""
        self._calib_samples.clear()
        self.calibrated = False

    def feed_calib_frame(self, gray) -> bool:
        """Feed one frame during calibration. Returns True if face found."""
        sample = self._measure(gray)
        if sample:
            self._calib_samples.append(sample)
            return True
        return False

    def finish_calibration(self) -> bool:
        """Compute baseline from collected samples. Returns True on success."""
        if len(self._calib_samples) < 5:
            return False
        if self.use_dlib:
            pitches = [s['pitch'] for s in self._calib_samples]
            yaws    = [s['yaw']   for s in self._calib_samples]
            self._baseline_pitch = float(np.median(pitches))
            self._baseline_yaw   = float(np.median(yaws))
        else:
            ratios = [s['ratio_y'] for s in self._calib_samples]
            self._baseline_ratio_y = float(np.median(ratios))
        self.calibrated = True
        self._window.clear()
        return True

   
    # Помощник по необработанным измерениям (без рисования рамки, без состояния)


    def _measure(self, gray) -> dict | None:
        """Return a dict of raw pose measurements for the first face found."""
        if self.use_dlib:
            faces = self.detector(gray)
            if not faces:
                return None
            lm = self.predictor(gray, faces[0])
            pitch, yaw = self._dlib_pose(lm)
            return {'pitch': pitch, 'yaw': yaw}
        else:
            frontal = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(frontal) == 0:
                return None
            (x, y, w, h) = frontal[0]
            fh = gray.shape[0]
            ratio_y = (y + h // 2) / fh
            return {'ratio_y': ratio_y}

    
    # Публичный API обнаружения
   

    def detect(self, frame, gray) -> tuple[bool, str]:
        if self.use_dlib:
            raw = self._detect_dlib(frame, gray)
        else:
            raw = self._detect_opencv(frame, gray)

        # --- no_face debounce -------------------------------------------------
        # Вычислите необходимые кадры из настроек (в режиме реального времени) или вернитесь к постоянному значению.
        nf_min = (
            max(1, int(self._settings.no_face_delay_sec * 30))
            if self._settings is not None
            else _NO_FACE_MIN_FRAMES
        )
        if raw == 'no_face':
            self._no_face_streak += 1
            if self._no_face_streak < nf_min:
                raw = self._last_face_reason   # pretend face is still visible
        else:
            self._no_face_streak = 0
            self._last_face_reason = raw

        # окно стабильности при голосовании большинством голосов
        self._window.append(raw)
        if len(self._window) > self._window_size:
            self._window.pop(0)

        stable = Counter(self._window).most_common(1)[0][0]
        return stable != 'focused', stable

    # помощник по извлечению позиции dlib
    

    @staticmethod
    def _dlib_pose(lm) -> tuple[float, float]:
        """Return (pitch, yaw) from 68-point landmarks."""
        nose_tip = (lm.part(30).x, lm.part(30).y)
        chin     = (lm.part(8).x,  lm.part(8).y)
        forehead = (lm.part(27).x, lm.part(27).y)
        l_outer  = lm.part(36)
        r_outer  = lm.part(45)
        eye_span = r_outer.x - l_outer.x + 1e-6
        face_cx  = (l_outer.x + r_outer.x) / 2
        yaw      = (nose_tip[0] - face_cx) / eye_span
        pitch    = (chin[1] - nose_tip[1]) / (nose_tip[1] - forehead[1] + 1e-6)
        return pitch, yaw

   
    # обнаружение dlib с относительными пороговыми значениями для калибровки
    

    def _detect_dlib(self, frame, gray) -> str:
        faces = self.detector(gray)
        if not faces:
            return 'no_face'

        lm = self.predictor(gray, faces[0])
        pitch, yaw = self._dlib_pose(lm)

        # Draw landmarks
        nose = (lm.part(30).x, lm.part(30).y)
        chin = (lm.part(8).x,  lm.part(8).y)
        fhd  = (lm.part(27).x, lm.part(27).y)
        lo   = lm.part(36); ro = lm.part(45)
        cv2.circle(frame, nose, 3, (0, 255, 0),   -1)
        cv2.circle(frame, chin, 3, (255, 0, 0),   -1)
        cv2.circle(frame, fhd,  3, (0, 165, 255), -1)
        cv2.circle(frame, (lo.x, lo.y), 3, (0, 255, 255), -1)
        cv2.circle(frame, (ro.x, ro.y), 3, (0, 255, 255), -1)

        # Считываются поля в реальном времени из настроек, если это доступно, в противном случае используются константы модуля
        _ym  = self._settings.yaw_margin        if self._settings else _YAW_MARGIN
        _pdm = self._settings.pitch_down_margin if self._settings else _PITCH_DOWN_MARGIN
        _pum = self._settings.pitch_up_margin   if self._settings else _PITCH_UP_MARGIN

        if self.calibrated and self._baseline_pitch is not None:
            bp = self._baseline_pitch
            by = self._baseline_yaw or 0.0
            if yaw - by > _ym:
                return 'turned_right'
            if yaw - by < -_ym:
                return 'turned_left'
            if pitch - bp > _pdm:
                return 'looking_up'
            if bp - pitch > _pum:
                return 'looking_down'
        else:
            # Некалиброванный: пропорционально изменяющиеся абсолютные пороговые значения
            yaw_abs   = 0.42 * (_ym  / _YAW_MARGIN)
            pd_abs    = 1.55 * (_pdm / _PITCH_DOWN_MARGIN)
            pu_abs    = 0.50 / (_pum  / _PITCH_UP_MARGIN)
            if yaw > yaw_abs:
                return 'turned_right'
            if yaw < -yaw_abs:
                return 'turned_left'
            if pitch > pd_abs:
                return 'looking_up'
            if pitch < pu_abs:
                return 'looking_down'

        return 'focused'

    
    # Обнаружение Haar с относительными пороговыми значениями для калибровки
    

    def _detect_opencv(self, frame, gray) -> str:
        fh, fw = frame.shape[:2]

        frontal = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(frontal) == 0:
            profile_l = self.profile_cascade.detectMultiScale(gray, 1.1, 4)
            if len(profile_l) > 0:
                px, py, pw, ph = profile_l[0]
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 165, 0), 2)
                return 'turned_left'

            gray_flip = cv2.flip(gray, 1)
            profile_r = self.profile_cascade.detectMultiScale(gray_flip, 1.1, 4)
            if len(profile_r) > 0:
                px, py, pw, ph = profile_r[0]
                px_orig = fw - px - pw
                cv2.rectangle(frame, (px_orig, py), (px_orig + pw, py + ph), (255, 165, 0), 2)
                return 'turned_right'

            return 'no_face'

        x, y, w, h = frontal[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray  = gray[y:y + int(h * 0.6), x:x + w]
        roi_color = frame[y:y + int(h * 0.6), x:x + w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        ratio_y = (y + h // 2) / fh

        # Нет глаз → возможно, повернутых под крайним углом.
        if len(eyes) == 0:
            if w / h < 0.75:
                return 'turned_left' if (x + w // 2) < fw // 2 else 'turned_right'
            if self.calibrated and self._baseline_ratio_y is not None:
                return 'looking_up' if ratio_y - self._baseline_ratio_y > 0.05 else 'looking_down'
            return 'looking_up' if ratio_y > 0.55 else 'looking_down'

        avg_ey          = sum(y + ey + eh // 2 for (_, ey, _, eh) in eyes) / len(eyes)
        eye_pos_in_face = (avg_ey - y) / h

        _rym = self._settings.ratio_y_margin if self._settings else _RATIO_Y_MARGIN
        if self.calibrated and self._baseline_ratio_y is not None:
            drift = ratio_y - self._baseline_ratio_y
            if drift > _rym or eye_pos_in_face < 0.30:
                return 'looking_down'
            if drift < -_rym or eye_pos_in_face > 0.56:
                return 'looking_up'
        else:
            # Некалиброванные абсолютные пороговые значения
            if eye_pos_in_face < 0.32 or ratio_y > 0.62:
                return 'looking_down'
            if eye_pos_in_face > 0.54 or ratio_y < 0.28:
                return 'looking_up'

        if len(eyes) == 1:
            eye_cx = x + eyes[0][0] + eyes[0][2] // 2
            return 'turned_left' if eye_cx < fw // 2 else 'turned_right'

        return 'focused'



# Настройки (общая изменяемая конфигурация)


class Settings:
    NOTIFY_VISUAL = 'visual'
    NOTIFY_SOUND  = 'sound'
    NOTIFY_BOTH   = 'both'

    def __init__(self):
        self.notification_mode = self.NOTIFY_BOTH
        self.sound_name = 'Мягкий сигнал'
        self.cooldown   = 4.0        # секунды между повторными предупреждениями

        # Чувствительность (поля увеличиваются → становятся менее чувствительными)
        # yaw горизонтальный поворот головы; pitch = вертикальный наклон
        self.yaw_margin:        float = _YAW_MARGIN
        self.pitch_down_margin: float = _PITCH_DOWN_MARGIN
        self.pitch_up_margin:   float = _PITCH_UP_MARGIN
        self.ratio_y_margin:    float = _RATIO_Y_MARGIN
        # Сколько секунд лица не было видно до появления предупреждения
        self.no_face_delay_sec: float = round(_NO_FACE_MIN_FRAMES / 30.0, 1)

    @property
    def sound_path(self) -> str:
        return os.path.join(SOUNDS_DIR, self.sound_name + '.wav')



# менеджер уведомлений


class NotificationManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._last_time = 0.0
        self._banner_active = False
        self._banner_cb: threading.Timer | None = None

    def trigger(self, root: tk.Tk, reason: str = 'no_face'):
        now = time.time()
        if now - self._last_time < self.settings.cooldown:
            return
        self._last_time = now

        mode = self.settings.notification_mode
        if mode in (Settings.NOTIFY_SOUND, Settings.NOTIFY_BOTH):
            threading.Thread(target=_play_wav, args=(self.settings.sound_path,), daemon=True).start()
        if mode in (Settings.NOTIFY_VISUAL, Settings.NOTIFY_BOTH):
            text = REASON_ALERT.get(reason, '⚠  Вы отвлеклись!')
            root.after(0, self._show_banner, root, text)

    def _show_banner(self, root: tk.Tk, text: str):
        if self._banner_active:
            return
        self._banner_active = True

        banner = tk.Toplevel(root)
        banner.overrideredirect(True)
        banner.attributes('-topmost', True)
        banner.configure(bg='#B71C1C')

        sw = root.winfo_screenwidth()
        w, h = 420, 70
        banner.geometry(f'{w}x{h}+{(sw - w) // 2}+20')

        tk.Label(
            banner, text=text,
            font=('Arial', 15, 'bold'), fg='white', bg='#B71C1C'
        ).pack(expand=True)

        def close():
            self._banner_active = False
            try:
                banner.destroy()
            except tk.TclError:
                pass

        self._banner_cb = threading.Timer(3.0, lambda: root.after(0, close))
        self._banner_cb.daemon = True
        self._banner_cb.start()
        banner.after(3000, close)



# окно статистики


class StatisticsWindow:
    def __init__(self, parent: tk.Tk, stats: Statistics):
        self.win = tk.Toplevel(parent)
        self.win.title('Статистика')
        self.win.resizable(False, False)
        self.win.grab_set()
        self.stats = stats
        self._build()
        self._update()

    def _build(self):
        win = self.win
        win.configure(bg='#1E1E2E', padx=28, pady=24)

        tk.Label(win, text='📊  Статистика сессии',
                 font=('Arial', 15, 'bold'), fg='white', bg='#1E1E2E').grid(
            row=0, column=0, columnspan=2, pady=(0, 18), sticky='w')

        labels = [
            'Длительность сессии:',
            'Время работы:',
            'Время отвлечений:',
            'Кол-во отвлечений:',
            'Фокус:',
        ]
        self._vars = [tk.StringVar() for _ in labels]

        for i, (lbl, var) in enumerate(zip(labels, self._vars)):
            tk.Label(win, text=lbl, font=('Arial', 11), fg='#AAAACC',
                     bg='#1E1E2E', anchor='w', width=24).grid(
                row=i + 1, column=0, sticky='w', pady=4)
            tk.Label(win, textvariable=var, font=('Arial', 11, 'bold'),
                     fg='white', bg='#1E1E2E', anchor='e').grid(
                row=i + 1, column=1, sticky='e', padx=(12, 0))

        ttk.Separator(win, orient='horizontal').grid(
            row=len(labels) + 1, column=0, columnspan=2, sticky='ew', pady=16)

        btn_frame = tk.Frame(win, bg='#1E1E2E')
        btn_frame.grid(row=len(labels) + 2, column=0, columnspan=2)

        tk.Button(btn_frame, text='Сбросить статистику',
                  font=('Arial', 10), fg='white', bg='#6C3483',
                  activebackground='#884EA0', relief='flat', padx=12, pady=6,
                  cursor='hand2', command=self._reset).pack(side='left', padx=6)
        tk.Button(btn_frame, text='Закрыть',
                  font=('Arial', 10), fg='white', bg='#2E4057',
                  activebackground='#3D5470', relief='flat', padx=12, pady=6,
                  cursor='hand2', command=self.win.destroy).pack(side='left', padx=6)

    def _reset(self):
        if messagebox.askyesno('Сброс', 'Сбросить всю статистику сессии?', parent=self.win):
            self.stats.reset()

    def _update(self):
        s = self.stats
        self._vars[0].set(s.fmt(s.session_seconds()))
        self._vars[1].set(s.fmt(s.work_seconds()))
        self._vars[2].set(s.fmt(s.current_distraction_seconds()))
        self._vars[3].set(str(s.distraction_count))
        self._vars[4].set(f'{s.focus_percent():.1f}%')
        try:
            self.win.after(1000, self._update)
        except tk.TclError:
            pass



# окно настроек

class SettingsWindow:
    def __init__(self, parent: tk.Tk, settings: Settings):
        self.win = tk.Toplevel(parent)
        self.win.title('Настройки')
        self.win.resizable(False, False)
        self.win.grab_set()
        self.settings = settings
        self._build()

    def _build(self):
        win = self.win
        win.configure(bg='#1E1E2E', padx=28, pady=24)

        tk.Label(win, text='⚙  Настройки',
                 font=('Arial', 15, 'bold'), fg='white', bg='#1E1E2E').pack(anchor='w', pady=(0, 18))

        # режим уведомления
        tk.Label(win, text='Способ уведомления', font=('Arial', 11, 'bold'),
                 fg='#AAAACC', bg='#1E1E2E').pack(anchor='w', pady=(0, 6))

        self._mode_var = tk.StringVar(value=self.settings.notification_mode)
        modes = [
            ('Визуальное (баннер)',      Settings.NOTIFY_VISUAL),
            ('Звуковое',                 Settings.NOTIFY_SOUND),
            ('Оба (визуальное + звук)',  Settings.NOTIFY_BOTH),
        ]
        for text, val in modes:
            tk.Radiobutton(
                win, text=text, variable=self._mode_var, value=val,
                font=('Arial', 11), fg='white', bg='#1E1E2E',
                selectcolor='#2E4057', activebackground='#1E1E2E',
                activeforeground='white'
            ).pack(anchor='w', padx=10)

        tk.Label(win, text='', bg='#1E1E2E').pack()

        # выбор мелодии
        tk.Label(win, text='Мелодия уведомления', font=('Arial', 11, 'bold'),
                 fg='#AAAACC', bg='#1E1E2E').pack(anchor='w', pady=(0, 6))

        sound_names = list(SOUND_DEFINITIONS.keys())
        self._sound_var = tk.StringVar(value=self.settings.sound_name)

        combo_frame = tk.Frame(win, bg='#1E1E2E')
        combo_frame.pack(anchor='w', padx=10, pady=(0, 6))

        self._combo = ttk.Combobox(combo_frame, textvariable=self._sound_var,
                                   values=sound_names, state='readonly', width=26,
                                   font=('Arial', 11))
        self._combo.pack(side='left', padx=(0, 8))

        tk.Button(combo_frame, text='▶ Тест', font=('Arial', 10),
                  fg='white', bg='#1F618D', activebackground='#2980B9',
                  relief='flat', padx=8, pady=4, cursor='hand2',
                  command=self._preview_sound).pack(side='left')

        tk.Label(win, text='', bg='#1E1E2E').pack()
        ttk.Separator(win, orient='horizontal').pack(fill='x', pady=8)

        # сладеры чувствительности
        tk.Label(win, text='Чувствительность обнаружения',
                 font=('Arial', 11, 'bold'), fg='#AAAACC', bg='#1E1E2E').pack(anchor='w', pady=(0, 6))

        def _slider_row(label_text, var, from_, to_, resolution=0.01):
            row = tk.Frame(win, bg='#1E1E2E')
            row.pack(fill='x', padx=10, pady=2)
            tk.Label(row, text=label_text, font=('Arial', 10), fg='white',
                     bg='#1E1E2E', width=26, anchor='w').pack(side='left')
            val_lbl = tk.Label(row, font=('Arial', 10), fg='#7EC8E3',
                               bg='#1E1E2E', width=5)
            val_lbl.pack(side='right')

            def _upd(v, lbl=val_lbl, vr=var):
                lbl.config(text=f'{float(v):.2f}')
            var.trace_add('write', lambda *_: _upd(var.get()))

            tk.Scale(row, variable=var, from_=from_, to=to_,
                     resolution=resolution, orient='horizontal',
                     length=200, bg='#1E1E2E', fg='white',
                     troughcolor='#2E4057', highlightthickness=0,
                     showvalue=False, command=_upd).pack(side='left', padx=(0, 4))
            _upd(var.get())

        
        self._yaw_var  = tk.DoubleVar(value=self.settings.yaw_margin)
        self._pdm_var  = tk.DoubleVar(value=self.settings.pitch_down_margin)
        self._pum_var  = tk.DoubleVar(value=self.settings.pitch_up_margin)
        self._nfd_var  = tk.DoubleVar(value=self.settings.no_face_delay_sec)

        _slider_row('Поворот головы (↔)',         self._yaw_var, 0.10, 0.70)
        _slider_row('Наклон вниз (↓)',             self._pdm_var, 0.20, 1.00)
        _slider_row('Наклон вверх (↑)',            self._pum_var, 0.20, 0.90)
        _slider_row('"Вышли из кадра" задержка, с', self._nfd_var, 0.3, 4.0, 0.1)

        tk.Label(win, text='Чем правее — тем больше свободы движения',
                 font=('Arial', 9), fg='#666688', bg='#1E1E2E').pack(anchor='w', padx=10)

        tk.Label(win, text='', bg='#1E1E2E').pack()
        ttk.Separator(win, orient='horizontal').pack(fill='x', pady=8)

        btn_frame = tk.Frame(win, bg='#1E1E2E')
        btn_frame.pack()

        tk.Button(btn_frame, text='Сохранить',
                  font=('Arial', 10), fg='white', bg='#1E8449',
                  activebackground='#27AE60', relief='flat', padx=14, pady=6,
                  cursor='hand2', command=self._save).pack(side='left', padx=6)
        tk.Button(btn_frame, text='Отмена',
                  font=('Arial', 10), fg='white', bg='#2E4057',
                  activebackground='#3D5470', relief='flat', padx=14, pady=6,
                  cursor='hand2', command=self.win.destroy).pack(side='left', padx=6)

    def _preview_sound(self):
        name = self._sound_var.get()
        path = os.path.join(SOUNDS_DIR, name + '.wav')
        threading.Thread(target=_play_wav, args=(path,), daemon=True).start()

    def _save(self):
        self.settings.notification_mode    = self._mode_var.get()
        self.settings.sound_name           = self._sound_var.get()
        self.settings.yaw_margin           = round(self._yaw_var.get(), 3)
        self.settings.pitch_down_margin    = round(self._pdm_var.get(), 3)
        self.settings.pitch_up_margin      = round(self._pum_var.get(), 3)
        self.settings.no_face_delay_sec    = round(self._nfd_var.get(), 2)
        self.settings.ratio_y_margin       = round(
            _RATIO_Y_MARGIN * (self._yaw_var.get() / _YAW_MARGIN), 3)
        self.win.destroy()



# Основное применение


class FocusMonitorApp:
    FRAME_W = 640
    FRAME_H = 480

    def __init__(self):
        _ensure_sounds()

        self.settings  = Settings()
        self.stats     = Statistics()
        self.detector  = FaceDetector(self.settings)
        self.notifier  = NotificationManager(self.settings)

        self._running   = False
        self._cap       = None
        self._cam_thread: threading.Thread | None = None
        self._latest_frame = None
        self._frame_lock   = threading.Lock()
        self._is_distracted = False
        self._distraction_reason = 'focused'

        # состояние калибровки (записывается из основного потока, считывается из потока камеры)
        self._calibrating    = False
        self._calib_end_time = 0.0
        self._calib_countdown = 0  # оставшиеся секунды, показанные на кадре

        self._build_ui()
        self._root.after(33, self._refresh_frame)

   
    # Построение пользовательского интерфейса
    

    def _build_ui(self):
        root = tk.Tk()
        root.title('Focus Monitor')
        root.configure(bg='#1E1E2E')
        root.resizable(False, False)
        self._root = root

        # верхний банер
        top = tk.Frame(root, bg='#13131F', pady=8, padx=16)
        top.pack(fill='x')

        tk.Label(top, text='🎯  Focus Monitor',
                 font=('Arial', 14, 'bold'), fg='white', bg='#13131F').pack(side='left')

        # строка с методом
        self._method_var = tk.StringVar(value=f'Метод: {self.detector.method_label}')
        tk.Label(top, textvariable=self._method_var,
                 font=('Arial', 9), fg='#888899', bg='#13131F').pack(side='right', padx=4)

        # панель подсказок по dlib (отображается только при установке dlib, но .dat отсутствует/поврежден)
        if self.detector.dlib_hint:
            hint_bar = tk.Frame(root, bg='#7D3A00', pady=6, padx=14)
            hint_bar.pack(fill='x')
            tk.Label(hint_bar, text='⚠  ' + self.detector.dlib_hint.split('\n')[0],
                     font=('Arial', 9), fg='#FFD580', bg='#7D3A00', anchor='w').pack(side='left')

            def _open_dat_folder():
                folder = os.path.dirname(_DAT_FILE)
                try:
                    if os.name == 'nt':
                        os.startfile(folder)
                    elif hasattr(os, 'uname') and os.uname().sysname == 'Darwin':
                        import subprocess
                        subprocess.Popen(['open', folder])
                    else:
                        import subprocess
                        subprocess.Popen(['xdg-open', folder])
                except Exception:
                    messagebox.showinfo('Папка для файла', folder, parent=root)

            tk.Button(hint_bar, text='📂 Открыть папку', font=('Arial', 9),
                      fg='white', bg='#5D6D00', activebackground='#7A8F00',
                      relief='flat', padx=6, pady=2, cursor='hand2',
                      command=_open_dat_folder).pack(side='right', padx=(4, 0))
            tk.Button(hint_bar, text='Подробнее', font=('Arial', 9),
                      fg='white', bg='#9E4A00', activebackground='#BF5A00',
                      relief='flat', padx=6, pady=2, cursor='hand2',
                      command=lambda: messagebox.showinfo(
                          'Как включить dlib', self.detector.dlib_hint, parent=root)
                      ).pack(side='right')

        # видео
        canvas_frame = tk.Frame(root, bg='#0D0D1A', padx=4, pady=4)
        canvas_frame.pack()

        self._canvas = tk.Canvas(canvas_frame, width=self.FRAME_W, height=self.FRAME_H,
                                 bg='#0D0D1A', highlightthickness=0)
        self._canvas.pack()

        # текст
        self._canvas.create_text(self.FRAME_W // 2, self.FRAME_H // 2,
                                 text='Нажмите «Старт» для начала мониторинга',
                                 fill='#555566', font=('Arial', 14),
                                 tags='placeholder')

        # банер со статусом
        self._status_var = tk.StringVar(value='Ожидание запуска...')
        status_bar = tk.Frame(root, bg='#2E4057', pady=6)
        status_bar.pack(fill='x')

        self._status_dot = tk.Label(status_bar, text='●', font=('Arial', 13),
                                    fg='#555566', bg='#2E4057')
        self._status_dot.pack(side='left', padx=(14, 4))

        tk.Label(status_bar, textvariable=self._status_var,
                 font=('Arial', 11), fg='white', bg='#2E4057').pack(side='left')

        # кнопки 
        ctrl = tk.Frame(root, bg='#1E1E2E', pady=12)
        ctrl.pack()

        self._start_btn = tk.Button(
            ctrl, text='▶  Старт',
            font=('Arial', 11, 'bold'), fg='white', bg='#1E8449',
            activebackground='#27AE60', relief='flat', padx=18, pady=8,
            cursor='hand2', command=self._toggle_monitoring)
        self._start_btn.pack(side='left', padx=8)

        self._calib_btn = tk.Button(
            ctrl, text='🎯  Калибровка',
            font=('Arial', 11), fg='white', bg='#7D6608',
            activebackground='#A07808', relief='flat', padx=14, pady=8,
            cursor='hand2', command=self._start_calibration, state='disabled')
        self._calib_btn.pack(side='left', padx=8)

        tk.Button(
            ctrl, text='📊  Статистика',
            font=('Arial', 11), fg='white', bg='#1F618D',
            activebackground='#2980B9', relief='flat', padx=14, pady=8,
            cursor='hand2', command=self._open_stats).pack(side='left', padx=8)

        tk.Button(
            ctrl, text='⚙  Настройки',
            font=('Arial', 11), fg='white', bg='#6C3483',
            activebackground='#884EA0', relief='flat', padx=14, pady=8,
            cursor='hand2', command=self._open_settings).pack(side='left', padx=8)

        root.protocol('WM_DELETE_WINDOW', self._on_close)

    
    # мониторинг
 

    def _toggle_monitoring(self):
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror('Ошибка', 'Не удалось открыть веб-камеру.')
            return
        self._cap = cap
        self._running = True
        self._is_distracted = False
        self._canvas.delete('placeholder')
        self._start_btn.configure(text='⏹  Стоп', bg='#922B21', activebackground='#CB4335')
        self._calib_btn.configure(state='normal')
        self._set_status('Мониторинг активен...', 'working')
        if not self.detector.calibrated:
            self._set_status('Рекомендуется нажать «Калибровка»', 'idle')
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

    def _stop(self):
        self._running = False
        self._calibrating = False
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._frame_lock:
            self._latest_frame = None
        if self._is_distracted:
            self.stats.on_distraction_end()
            self._is_distracted = False
        self._start_btn.configure(text='▶  Старт', bg='#1E8449', activebackground='#27AE60')
        self._calib_btn.configure(state='disabled')
        self._set_status('Мониторинг остановлен', 'idle')
        self._canvas.delete('all')
        self._canvas.create_text(self.FRAME_W // 2, self.FRAME_H // 2,
                                 text='Нажмите «Старт» для начала мониторинга',
                                 fill='#555566', font=('Arial', 14),
                                 tags='placeholder')

    
    # калибовка
 

    def _start_calibration(self):
        if not self._running or self._calibrating:
            return
        self.detector.start_calibration()
        self._calibrating    = True
        self._calib_end_time = time.time() + FaceDetector.CALIB_SECS
        self._calib_btn.configure(state='disabled')
        self._set_status('Калибровка: смотрите прямо в экран...', 'idle')

    
    # Цикл работы камеры (выполняется в фоновом режиме)
   

    def _camera_loop(self):
        while self._running and self._cap and self._cap.isOpened():
            ok, frame = self._cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # режим калиброки
            if self._calibrating:
                remaining = self._calib_end_time - time.time()
                self._calib_countdown = max(0, math.ceil(remaining))
                self.detector.feed_calib_frame(gray)

                if remaining <= 0:
                    self._calibrating = False
                    ok_calib = self.detector.finish_calibration()
                    self._root.after(0, self._on_calib_done, ok_calib)

                self._draw_overlay_calib(frame, self._calib_countdown)
                with self._frame_lock:
                    self._latest_frame = frame.copy()
                continue

            # нормальная детекция
            is_dist, reason = self.detector.detect(frame, gray)
            self._distraction_reason = reason

            if is_dist and not self._is_distracted:
                self.stats.on_distraction_start()
                self._is_distracted = True
            elif not is_dist and self._is_distracted:
                self.stats.on_distraction_end()
                self._is_distracted = False

            if is_dist:
                self.notifier.trigger(self._root, reason)

            self._draw_overlay(frame, is_dist, reason)

            with self._frame_lock:
                self._latest_frame = frame.copy()

    def _on_calib_done(self, success: bool):
        self._calib_btn.configure(state='normal')
        if success:
            self._set_status('Калибровка завершена — мониторинг активен', 'working')
        else:
            self._set_status('Калибровка не удалась: лицо не обнаружено', 'distracted')

    # Цвета наложения в зависимости от причины (BGR)
    _REASON_COLOUR = {
        'focused':      (0,   140,  0),
        'looking_down': (0,   0,    180),
        'looking_up':   (180, 80,   0),
        'turned_left':  (150, 0,    150),
        'turned_right': (150, 0,    150),
        'no_face':      (60,  60,   60),
    }

   
    # Средство визуализации текста на основе PIL 
 

    _font_cache: dict = {}

    @classmethod
    def _get_font(cls, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (size, bold)
        if key in cls._font_cache:
            return cls._font_cache[key]
        candidates = []
        if bold:
            candidates += [
                'arialbd.ttf', 'Arial Bold.ttf',
                'C:/Windows/Fonts/arialbd.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
            ]
        candidates += [
            'arial.ttf', 'Arial.ttf',
            'C:/Windows/Fonts/arial.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/System/Library/Fonts/Supplemental/Arial.ttf',
        ]
        font = None
        for name in candidates:
            try:
                font = ImageFont.truetype(name, size)
                break
            except Exception:
                pass
        if font is None:
            font = ImageFont.load_default()
        cls._font_cache[key] = font
        return font

    @classmethod
    def _draw_text_pil(cls, frame, items):
        """Draw list of (text, x, y, size, color_rgb, bold) onto a BGR frame in-place."""
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for text, x, y, size, color_rgb, bold in items:
            draw.text((x, y), text, font=cls._get_font(size, bold), fill=color_rgb)
        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------

    def _draw_overlay(self, frame, is_distracted: bool, reason: str):
        h, w = frame.shape[:2]
        colour = self._REASON_COLOUR.get(reason, (0, 0, 180))

        # Тонкая цветная полоска по верхнему краю, только когда отвлекаешься.
        if is_distracted:
            cv2.rectangle(frame, (0, 0), (w, 6), colour, -1)

        # Вверху слева: метод + статус калибровки
        calib_label = 'Откалибровано' if self.detector.calibrated else 'Без калибровки'
        calib_color = (80, 200, 80) if self.detector.calibrated else (80, 180, 220)
        self._draw_text_pil(frame, [(
            f'{self.detector.method_label}  |  {calib_label}',
            8, 8, 13, calib_color, False,
        )])

    def _draw_overlay_calib(self, frame, countdown: int):
        """Calibration-mode overlay: dim frame, show countdown and instruction."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 80, 120), -1)
        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

        # Круг обратного отсчета (нарисован с помощью cv2, текст не требуется)
        cx, cy, r = w // 2, h // 2 + 60, 36
        cv2.circle(frame, (cx, cy), r, (255, 220, 50), 3)

        self._draw_text_pil(frame, [
            ('КАЛИБРОВКА',             w // 2 - 120, h // 2 - 80, 30, (255, 220, 50),  True),
            ('Смотрите прямо в экран', w // 2 - 165, h // 2 - 22, 20, (255, 255, 255), False),
            (str(countdown),           cx - 14,      cy - 20,     32, (255, 220, 50),  True),
        ])

    
    # Обновление кадра (выполняется в основном потоке tkinter через after())
   

    def _refresh_frame(self):
        with self._frame_lock:
            frame = self._latest_frame

        if frame is not None:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb).resize((self.FRAME_W, self.FRAME_H))
            photo = ImageTk.PhotoImage(img)
            self._canvas.delete('frame')
            self._canvas.create_image(0, 0, anchor='nw', image=photo, tags='frame')
            self._canvas._photo = photo  # keep reference

            reason = self._distraction_reason
            if self._is_distracted:
                label = REASON_LABELS.get(reason, 'Отвлечение')
                self._set_status(f'Внимание: {label}', 'distracted')
            else:
                self._set_status('Вы сосредоточены — отличная работа!', 'working')

        try:
            self._root.after(33, self._refresh_frame)
        except tk.TclError:
            pass

  
    # Помощники по статусу
    

    def _set_status(self, text: str, state: str):
        colors = {'working': '#27AE60', 'distracted': '#E74C3C', 'idle': '#555566'}
        self._status_var.set(text)
        self._status_dot.configure(fg=colors.get(state, '#555566'))

    
    # диалоговые окна
    

    def _open_stats(self):
        StatisticsWindow(self._root, self.stats)

    def _open_settings(self):
        SettingsWindow(self._root, self.settings)

    
    # Lifecycle
    

    def _on_close(self):
        self._stop()
        self._root.destroy()

    def run(self):
        self._root.mainloop()



# EntrY


if __name__ == '__main__':
    app = FocusMonitorApp()
    app.run()
