# ml_app/views_part1.py
import glob
from django.shortcuts import render, redirect
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
import os
import numpy as np
import cv2
from PIL import Image as pImage
import time
from django.conf import settings
from .forms import VideoUploadForm

from blazeface.blazeface import BlazeFace 

from helpers.read_video_1 import VideoReader
from helpers.face_extract_1 import FaceExtractor
from pytorchcv.model_provider import get_model
from concurrent.futures import ThreadPoolExecutor

index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"
cuda_tempalte_name = "cuda.html"

IM_SIZE = 150
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NORMALIZE_TRANSFORM = Normalize(MEAN, STD)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# BlazeFace setup
FACEDET = BlazeFace().to(DEVICE)
blazeface_model_path = os.path.join(settings.BASE_DIR, 'blazeface', 'blazeface.pth')
anchors_path = os.path.join(settings.BASE_DIR, 'blazeface', 'anchors.npy')
FACEDET.load_weights(blazeface_model_path)
FACEDET.load_anchors(anchors_path)
_ = FACEDET.train(False)

VIDEO_READER = VideoReader()
FRAMES_PER_VIDEO = 12  # Or get from frontend
VIDEO_READ_FN = lambda x: VIDEO_READER.read_frames(x, num_frames=FRAMES_PER_VIDEO)
FACE_EXTRACTOR = FaceExtractor(VIDEO_READ_FN, FACEDET)

def index(request):
    return render(request, index_template_name)

def about(request):
    return render(request, about_template_name)

def predict_page(request):
    return render(request, predict_template_name)

def cuda_full(request):
    return render(request, "cuda.html")

def handler404(request, exception):
    return render(request, '404.html', status=404)

def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size
    return cv2.resize(img, (w, h), interpolation=resample)

def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

def load_xception_model(model_path):
    model = get_model("xception", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1])
    model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

    class Head(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.f = nn.Flatten()
            self.l = nn.Linear(in_f, 512)
            self.d = nn.Dropout(0.5)
            self.o = nn.Linear(512, out_f)
            self.b1 = nn.BatchNorm1d(in_f)
            self.b2 = nn.BatchNorm1d(512)
            self.r = nn.ReLU()

        def forward(self, x):
            x = self.f(x)
            x = self.b1(x)
            x = self.d(x)
            x = self.l(x)
            x = self.r(x)
            x = self.b2(x)
            x = self.d(x)
            return self.o(x)

    class FCN(nn.Module):
        def __init__(self, base, in_f):
            super().__init__()
            self.base = base
            self.h1 = Head(in_f, 1)

        def forward(self, x):
            x = self.base(x)
            return self.h1(x)

    model = FCN(model, 2048).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def load_models(model_dir):
    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith(".pth"):
            model_path = os.path.join(model_dir, filename)
            model = load_xception_model(model_path)
            models.append(model)
    return models

def predict_on_video(video_path, model_dir, batch_size=FRAMES_PER_VIDEO):
    try:
        models = load_models(model_dir)
        faces = FACE_EXTRACTOR.process_video(video_path)
        FACE_EXTRACTOR.keep_only_best_face(faces)

        if not faces:
            return 0.5

        x = np.zeros((batch_size, IM_SIZE, IM_SIZE, 3), dtype=np.uint8)
        n = 0
        for frame_data in faces:
            for face in frame_data["faces"]:
                resized_face = isotropically_resize_image(face, IM_SIZE)
                resized_face = make_square_image(resized_face)

                if n < batch_size:
                    x[n] = resized_face
                    n += 1
                else:
                    print(f"WARNING: have {n} faces but batch size is {batch_size}")
                    break
            if n >= batch_size:
                break

        if n == 0:
            return 0.5

        x = torch.tensor(x[:n], device=DEVICE).float().permute((0, 3, 1, 2))
        for i in range(n):
            x[i] = NORMALIZE_TRANSFORM(x[i] / 255.)

        with torch.no_grad():
            all_predictions = []
            for model in models:
                y_pred = model(x)
                y_pred = torch.sigmoid(y_pred.squeeze())
                all_predictions.append(y_pred[:n].mean().item())

            return sum(all_predictions) / len(all_predictions)

    except Exception as e:
        print(f"Prediction error on video {video_path}: {e}")
        return 0.5

def predict_on_video_set(video_path, models):
    return predict_on_video(video_path, os.path.join(settings.BASE_DIR, 'models'))

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'}

def allowed_video_file(filename):
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def index(request):
    if request.method == 'GET':
        return render(request, index_template_name, {"form": VideoUploadForm()})
    else:
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = form.cleaned_data['upload_video_file']
            ext = video_file.name.split('.')[-1]
            seq_len = form.cleaned_data['sequence_length']
            if video_file.content_type.split('/')[0] not in settings.CONTENT_TYPES:
                form.add_error("upload_video_file", "Invalid content type")
                return render(request, index_template_name, {"form": form})
            if video_file.size > settings.MAX_UPLOAD_SIZE:
                form.add_error("upload_video_file", "Maximum file size exceeded")
                return render(request, index_template_name, {"form": form})
            if seq_len <= 0:
                form.add_error("sequence_length", "Sequence length must be positive")
                return render(request, index_template_name, {"form": form})
            if not allowed_video_file(video_file.name):
                form.add_error("upload_video_file", "Invalid video format")
                return render(request, index_template_name, {"form": form})

            saved_file = f'uploaded_file_{int(time.time())}.{ext}'
            upload_path = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_file)
            with open(upload_path, 'wb') as dest:
                for chunk in video_file.chunks():
                    dest.write(chunk)

            request.session['file_name'] = upload_path
            request.session['sequence_length'] = seq_len
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": form})

def predict_page(request):
    if request.method != "GET" or 'file_name' not in request.session:
        return redirect("ml_app:home")

    video_file = request.session['file_name']
    model_dir = os.path.join(settings.BASE_DIR, 'models')
    models = load_models(model_dir)

    # Process video and extract frames/faces
    faces_data = FACE_EXTRACTOR.process_video(video_file)
    FACE_EXTRACTOR.keep_only_best_face(faces_data)

    original_frames, original_frame_indices = VIDEO_READER.read_frames(video_file, num_frames=FRAMES_PER_VIDEO)

    split_frame_paths = []
    cropped_face_paths = []
    boxed_frame_paths = []

    if faces_data and original_frames is not None:
        for i, frame_data in enumerate(faces_data):
            if i >= len(original_frames):
                break

            original_frame = original_frames[i]
            frame_filename = f"split_frame_{i}.jpg"
            frame_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', frame_filename)
            cv2.imwrite(frame_path, cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))
            split_frame_paths.append(frame_filename)

            if frame_data['faces']:
                face = frame_data['faces'][0]
                face_filename = f"cropped_face_{i}.jpg"
                face_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', face_filename)
                cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                cropped_face_paths.append(face_filename)

                # Draw bounding box (red for deepfake, green for real)
                prediction = predict_on_video(video_file, model_dir)
                color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

                # Use frame data coords if available.
                if 'coords' in frame_data and frame_data['coords']:
                    x, y, w, h = frame_data['coords'][0]
                    print(f"Frame {i}: Coords - x: {x}, y: {y}, w: {w}, h: {h}")  # Added print statement
                    boxed_frame = original_frame.copy()
                    cv2.rectangle(boxed_frame, (x, y), (x + w, y + h), color, 2)
                    boxed_filename = f"boxed_frame_{i}.jpg"
                    boxed_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', boxed_filename)
                    cv2.imwrite(boxed_path, cv2.cvtColor(boxed_frame, cv2.COLOR_RGB2BGR))
                    boxed_frame_paths.append(boxed_filename)
                else:
                    print(f"Frame {i}: Analyzing")  # Added print statement
                    boxed_frame_paths.append(split_frame_paths[-1])  # Use original frame if no coords
            else:
                cropped_face_paths.append(None)
                boxed_frame_paths.append(split_frame_paths[-1])

    prediction = predict_on_video(video_file, model_dir)

    # Prepare data for template
    context = {
        'prediction': prediction,
        'split_frames': split_frame_paths,
        'cropped_faces': cropped_face_paths,
        'boxed_frames': boxed_frame_paths,
        'MEDIA_URL': settings.MEDIA_URL,
        'original_video': os.path.basename(video_file),
    }

    # Convert prediction into real or fake
    if prediction > 0.5:
        context['output'] = "DEEPFAKE"
        context['confidence'] = round(prediction * 100, 2)
    else:
        context['output'] = "REAL"
        context['confidence'] = round((1 - prediction) * 100, 2)

    return render(request, predict_template_name, context)