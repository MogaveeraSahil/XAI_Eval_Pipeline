import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import os
import pandas as pd

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    img = img.astype(np.uint8)
    
    overlay = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    return overlay

def load_image(path, img_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return img_array

def integrated_gradients(model, img, steps=50):
    
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    baseline = tf.zeros_like(img)
    
    # Generate interpolated images
    alphas = tf.linspace(0.0, 1.0, steps+1)
    alphas = tf.reshape(alphas, (-1, 1, 1, 1))
    
    interpolated = baseline + alphas * (img - baseline)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated)
        loss = preds[:, 0]
    
    grads = tape.gradient(loss, interpolated)
    
    # Average gradients
    avg_grads = tf.reduce_mean(grads, axis=0)
    
    integrated_grads = (img - baseline)[0] * avg_grads
    
    heatmap = tf.reduce_sum(tf.abs(integrated_grads), axis=-1)
    
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()

def smoothgrad(model, img, num_samples=20, noise_level=0.1):
    
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    
    grads_list = []
    
    for _ in range(num_samples):
        noise = tf.random.normal(shape=tf.shape(img), stddev=noise_level)
        noisy_img = img + noise
        
        with tf.GradientTape() as tape:
            tape.watch(noisy_img)
            preds = model(noisy_img)
            loss = preds[:, 0]
        
        grads = tape.gradient(loss, noisy_img)
        grads_list.append(grads)
    
    avg_grads = tf.reduce_mean(tf.stack(grads_list), axis=0)
    
    heatmap = tf.reduce_mean(tf.abs(avg_grads[0]), axis=-1)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()

def gradcam_plus_plus(img_array, model, last_conv_layer_name):
    
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.outputs[0]]
    )
    
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    
    # First, second, third derivatives
    first = grads
    second = grads ** 2
    third = grads ** 3
    
    global_sum = tf.reduce_sum(conv_outputs, axis=(1,2))
    
    alpha_num = second
    alpha_denom = 2 * second + third * global_sum[:, None, None, :]
    alpha_denom = tf.where(alpha_denom != 0, alpha_denom, tf.ones_like(alpha_denom))
    
    alphas = alpha_num / alpha_denom
    
    weights = tf.reduce_sum(alphas * tf.nn.relu(first), axis=(1,2))
    
    cam = tf.reduce_sum(weights[:, None, None, :] * conv_outputs, axis=-1)
    
    heatmap = tf.maximum(cam[0], 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()

def smooth(hm):
    return cv2.GaussianBlur(hm, (11,11), 0)

def get_central_slice(patient_path):
    
    flair_path = [f for f in os.listdir(patient_path) if "flair" in f][0]
    seg_path   = [f for f in os.listdir(patient_path) if "seg" in f][0]
    
    flair = nib.load(os.path.join(patient_path, flair_path)).get_fdata()
    mask  = nib.load(os.path.join(patient_path, seg_path)).get_fdata()
    
    tumor_slices = [i for i in range(mask.shape[2]) if np.sum(mask[:,:,i]) > 0]
    
    mid = tumor_slices[len(tumor_slices)//2]
    
    img = flair[:,:,mid]
    gt  = (mask[:,:,mid] > 0).astype(np.uint8)
    
    # normalize
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 255).astype(np.uint8)
    
    img = cv2.resize(img, (224,224))
    gt  = cv2.resize(gt, (224,224), interpolation=cv2.INTER_NEAREST)
    
    # convert to 3-channel
    img_3 = np.stack([img]*3, axis=-1)
    
    return img_3, gt

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum((y_true + y_pred) > 0)
    return intersection / (union + 1e-8)


def heatmap_to_mask(heatmap, thresh=0.4):
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return (heatmap > thresh).astype(np.uint8)

def get_score(model, img, preprocess_fn):

    img_batch = np.expand_dims(img, axis=0)

    if preprocess_fn is not None:
        img_batch = preprocess_fn(img_batch)

    pred = model.predict(img_batch, verbose=0)

    return float(pred[0][0])

def prepare_heatmap(heatmap):
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return heatmap

def normalize_heatmap(hm):
    hm = hm - np.min(hm)
    hm = hm / (np.max(hm) + 1e-8)
    return hm

def compute_auc(scores):
    scores = np.array(scores)
    x = np.linspace(0, 1, len(scores))
    return np.trapz(scores, x)

def deletion_pixel_black(model, img, heatmap, preprocess_fn, steps=20):
    
    heatmap = normalize_heatmap(cv2.resize(heatmap, (224,224)))
    
    h, w, _ = img.shape
    indices = np.argsort(-heatmap.flatten())
    step_size = (h*w)//steps
    
    img_temp = img.copy().reshape(-1,3)
    scores = []
    
    for i in range(steps):
        idx = indices[:(i+1)*step_size]
        img_temp[idx] = 0
        
        current = img_temp.reshape(h,w,3)
        scores.append(get_score(model, current, preprocess_fn))
    
    return scores

def insertion_pixel_black(model, img, heatmap, preprocess_fn, steps=20):
    
    heatmap = normalize_heatmap(cv2.resize(heatmap, (224,224)))
    
    h, w, _ = img.shape
    indices = np.argsort(-heatmap.flatten())
    step_size = (h*w)//steps
    
    img_temp = np.zeros_like(img).reshape(-1,3)
    original = img.reshape(-1,3)
    
    scores = []
    
    for i in range(steps):
        idx = indices[:(i+1)*step_size]
        img_temp[idx] = original[idx]
        
        current = img_temp.reshape(h,w,3)
        scores.append(get_score(model, current, preprocess_fn))
    
    return scores

def deletion_pixel_blur(model, img, heatmap, preprocess_fn, steps=20):
    
    heatmap = normalize_heatmap(cv2.resize(heatmap, (224,224)))
    
    h, w, _ = img.shape
    blur = cv2.GaussianBlur(img, (51,51), 0)
    
    indices = np.argsort(-heatmap.flatten())
    step_size = (h*w)//steps
    
    img_temp = img.copy().reshape(-1,3)
    blur_flat = blur.reshape(-1,3)
    
    scores = []
    
    for i in range(steps):
        idx = indices[:(i+1)*step_size]
        img_temp[idx] = blur_flat[idx]
        
        current = img_temp.reshape(h,w,3)
        scores.append(get_score(model, current, preprocess_fn))
    
    return scores

def insertion_pixel_blur(model, img, heatmap, preprocess_fn, steps=20):
    
    heatmap = normalize_heatmap(cv2.resize(heatmap, (224,224)))
    
    h, w, _ = img.shape
    blur = cv2.GaussianBlur(img, (51,51), 0)
    
    indices = np.argsort(-heatmap.flatten())
    step_size = (h*w)//steps
    
    img_temp = blur.copy().reshape(-1,3)
    original = img.reshape(-1,3)
    
    scores = []
    
    for i in range(steps):
        idx = indices[:(i+1)*step_size]
        img_temp[idx] = original[idx]
        
        current = img_temp.reshape(h,w,3)
        scores.append(get_score(model, current, preprocess_fn))
    
    return scores

def deletion_region_blur(model, img, heatmap, preprocess_fn, steps=20, patch=8):
    
    heatmap = normalize_heatmap(cv2.resize(heatmap, (224,224)))
    
    h, w = heatmap.shape
    blur = cv2.GaussianBlur(img, (51,51), 0)
    
    # Compute patch importance
    patches = []
    for i in range(0, h, patch):
        for j in range(0, w, patch):
            score = np.mean(heatmap[i:i+patch, j:j+patch])
            patches.append((score, i, j))
    
    patches.sort(reverse=True)
    step_size = len(patches)//steps
    
    img_temp = img.copy()
    scores = []
    
    for i in range(steps):
        for k in range((i+1)*step_size):
            _, x, y = patches[k]
            img_temp[x:x+patch, y:y+patch] = blur[x:x+patch, y:y+patch]
        
        scores.append(get_score(model, img_temp, preprocess_fn))
    
    return scores

def insertion_region_blur(model, img, heatmap, preprocess_fn, steps=20, patch=8):
    
    heatmap = normalize_heatmap(cv2.resize(heatmap, (224,224)))
    
    h, w = heatmap.shape
    blur = cv2.GaussianBlur(img, (51,51), 0)
    
    patches = []
    for i in range(0, h, patch):
        for j in range(0, w, patch):
            score = np.mean(heatmap[i:i+patch, j:j+patch])
            patches.append((score, i, j))
    
    patches.sort(reverse=True)
    step_size = len(patches)//steps
    
    img_temp = blur.copy()
    scores = []
    
    for i in range(steps):
        for k in range((i+1)*step_size):
            _, x, y = patches[k]
            img_temp[x:x+patch, y:y+patch] = img[x:x+patch, y:y+patch]
        
        scores.append(get_score(model, img_temp, preprocess_fn))
    
    return scores

def evaluate_setting(model, img, methods, del_fn, ins_fn, preprocess_fn, name):
    
    records = []
    
    for m, hm in methods.items():
        
        del_scores = del_fn(model, img, hm, preprocess_fn)
        ins_scores = ins_fn(model, img, hm, preprocess_fn)
        
        records.append({
            "Method": m,
            "Deletion_AUC": compute_auc(del_scores),
            "Insertion_AUC": compute_auc(ins_scores)
        })
    
    df = pd.DataFrame(records)
    print(f"\n{name} Results:\n")
    print(df)
    
    return df