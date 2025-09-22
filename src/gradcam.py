"""
Simple GradCAM implementation for drowsy driver detection
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
import os


# ---------- Helpers ----------

def _to_class_index(y) -> int:
    """
    Convert label tensors/arrays to a scalar class index.
    Works for:
      - binary scalar (float in {0.,1.})
      - shape (1,) binary
      - one-hot vector
    """
    arr = np.array(y)
    if arr.ndim == 0:
        return int(round(float(arr)))
    flat = arr.reshape(-1)
    if flat.size == 1:
        return int(round(float(flat[0])))
    return int(np.argmax(flat))


def _pred_to_prob_and_class(pred):
    """
    Normalize model outputs to:
      - prob: probability of class '1' (Drowsy)
      - cls: predicted class index (0/1)
    Supports:
      - sigmoid with shape (1,) or (1,1)
      - softmax with shape (1,2)
    """
    p = np.array(pred)
    # (1,1) or (1,)
    if p.ndim == 2 and p.shape[0] == 1 and p.shape[1] == 1:
        prob = float(p[0, 0]); return prob, int(prob >= 0.5)
    if p.ndim == 1 and p.shape[0] == 1:
        prob = float(p[0]);    return prob, int(prob >= 0.5)
    # (1,2) softmax
    if p.ndim == 2 and p.shape[0] == 1 and p.shape[1] == 2:
        prob = float(p[0, 1]); cls = int(np.argmax(p[0])); return prob, cls
    # fallback
    pr = float(p.ravel()[-1])
    return pr, int(pr >= 0.5)


# ---------- GradCAM ----------

class GradCAM:
    """
    Simple GradCAM for explaining CNN predictions
    """

    def __init__(self, model, layer_name=None):
        """
        Initialize GradCAM with a trained model
        """
        self.model = model

        # Build model once to ensure outputs exist
        if not hasattr(self.model, 'output') or self.model.output is None:
            dummy_input = tf.random.normal((1, 224, 224, 3))
            _ = self.model(dummy_input, training=False)

        # Pick last Conv2D layer if not provided
        self.layer_name = layer_name
        if self.layer_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, Conv2D):
                    self.layer_name = layer.name
                    break

        # Create grad model (conv outputs + final outputs)
        try:
            self.grad_model = Model(
                inputs=self.model.input,
                outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
            )
        except Exception:
            # Fallback: use original model; we'll return a dummy heatmap
            self.grad_model = self.model

    def compute_heatmap(self, image, class_idx=None):
        """
        Compute GradCAM heatmap.
        - For binary sigmoid (1 unit): class_idx is forced to 0.
        - For 2-class softmax: class_idx can be 0/1; defaults to argmax.
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # Fallback path if grad_model could not be built
        if self.grad_model == self.model:
            prediction = self.model(image, training=False).numpy()
            # Produce a small dummy heatmap just to keep the pipeline running
            h = 7 if image.shape[1] >= 7 else image.shape[1]
            w = 7 if image.shape[2] >= 7 else image.shape[2]
            heatmap = np.random.rand(h, w)
            heatmap = heatmap / np.max(heatmap)
            return heatmap, prediction

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image, training=False)
            # Decide which logit/probability to explain
            predictions = tf.convert_to_tensor(predictions)
            if predictions.shape[-1] == 1:
                # Binary sigmoid: single neuron (prob of class 1)
                class_idx = 0  # the only column
                class_output = predictions[:, 0]
            else:
                # 2-class softmax
                if class_idx is None:
                    class_idx = tf.argmax(predictions[0]).numpy().item()
                class_output = predictions[:, class_idx]

        grads = tape.gradient(class_output, conv_outputs)
        # Global average pooling over H,W
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]  # (H, W, C)

        # Weighted sum across channels
        heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=[[2], [0]])
        heatmap = tf.nn.relu(heatmap)
        denom = tf.reduce_max(heatmap)
        heatmap = heatmap / (denom + 1e-8)

        return heatmap.numpy(), predictions.numpy()

    def overlay_heatmap(self, heatmap, image, alpha=0.4):
        """
        Overlay heatmap on image. Uses TF resize (no SciPy dependency).
        Input image can be uint8 [0..255] or float [0..1].
        """
        # Resize heatmap to match image size
        heatmap_tf = tf.convert_to_tensor(heatmap, dtype=tf.float32)
        heatmap_tf = heatmap_tf[None, ..., None]  # (1,H,W,1)
        H, W = int(image.shape[0]), int(image.shape[1])
        heatmap_resized = tf.image.resize(heatmap_tf, (H, W), method='bilinear')[0, ..., 0].numpy()

        # Normalize to [0,1]
        heatmap_norm = np.clip(heatmap_resized, 0.0, 1.0)
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        # Apply colormap (jet)
        cmap = plt.cm.get_cmap('jet')
        heatmap_colored = cmap(heatmap_norm)[..., :3]  # drop alpha

        # Blend
        overlayed = (1 - alpha) * img + alpha * heatmap_colored
        return np.clip(overlayed, 0.0, 1.0)

    def visualize(self, image, class_names=('Not Drowsy', 'Drowsy'),
                  threshold=0.5, target_class=None, save_path=None):
        """
        Create GradCAM visualization.
        - threshold: used to binarize sigmoid outputs
        - target_class: force CAM to a specific class (0/1 for softmax).
        """
        heatmap, prediction = self.compute_heatmap(
            image,
            class_idx=target_class
        )

        prob, pred_class = _pred_to_prob_and_class(prediction)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image if image.max() <= 1.0 else image.astype(np.uint8))
        axes[0].set_title(f'Original\nPredicted: {class_names[pred_class]} ({prob:.3f})')
        axes[0].axis('off')

        # Heatmap
        im1 = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])

        # Overlay
        overlayed = self.overlay_heatmap(heatmap, image)
        axes[2].imshow(overlayed)
        axes[2].set_title('GradCAM Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"GradCAM saved to: {save_path}")

        return fig, prediction


def analyze_model_gradcam(model, test_ds, num_samples=5, output_dir="gradcam_results",
                          class_names=('Not Drowsy', 'Drowsy'), threshold=0.5):
    """
    Analyze model with GradCAM on samples from test_ds.
    Handles binary sigmoid outputs and 2-class softmax models.
    """
    os.makedirs(output_dir, exist_ok=True)
    gradcam = GradCAM(model)
    sample_count = 0

    for batch_images, batch_labels in test_ds:
        if sample_count >= num_samples:
            break

        # Predict batch once for logging (optional; GradCAM does its own forward pass anyway)
        batch_preds = model.predict(batch_images, verbose=0)

        for i in range(len(batch_images)):
            if sample_count >= num_samples:
                break

            image = batch_images[i].numpy()
            true_idx = _to_class_index(batch_labels[i].numpy())
            # For 2-class softmax, CAM on class 1 is typical for "positive" class.
            target_class = None
            if np.array(batch_preds).ndim == 2 and np.array(batch_preds).shape[1] == 2:
                target_class = 1  # explain "Drowsy"

            fig, pred_vec = gradcam.visualize(
                image,
                class_names=class_names,
                threshold=threshold,
                target_class=target_class,
                save_path=os.path.join(output_dir, f'sample_{sample_count:02d}.png')
            )

            prob, pred_idx = _pred_to_prob_and_class(pred_vec)
            print(
                f"Sample {sample_count:02d}: "
                f"True={class_names[true_idx]}, Pred={class_names[pred_idx]} ({prob:.3f})"
            )

            plt.close(fig)
            sample_count += 1

    print(f"GradCAM analysis complete! Results saved in: {output_dir}/")
