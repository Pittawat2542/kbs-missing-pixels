import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from os.path import join

NUM_INP_NODES = 60
NUM_OUT_NODES = 4
BLOCK_SIZE = 8
M1 = 3
M2 = 4
C1 = 0
C2 = 7


def is_missing_pixel(r, c):
    return (r >= M1 and c >= M1 and r <= M2 and c <= M2)


def do_parse_img(img, inps, tgts):
    num_rows = img.shape[0]
    num_cols = img.shape[1]

    inp = np.zeros((NUM_INP_NODES))
    tgt = np.zeros((NUM_OUT_NODES))

    for r0 in range(0, num_rows, BLOCK_SIZE):
        for c0 in range(0, num_cols, BLOCK_SIZE):
            pos1 = 0
            pos2 = 0
            for r in range(BLOCK_SIZE):
                for c in range(BLOCK_SIZE):
                    assert (r0 + r < num_rows)
                    assert (c0 + c < num_cols)

                    if is_missing_pixel(r, c):
                        tgt[pos2] = img[r0 + r, c0 + c]
                        pos2 += 1
                    else:
                        inp[pos1] = img[r0 + r, c0 + c]
                        pos1 += 1

            inps.append(inp.copy() / 255.0)
            tgts.append(tgt.copy() / 255.0)


def imshow(img, cmap=None, vmin=0, vmax=255, frameon=False, dpi=72):
    fig = plt.figure(figsize=[img.shape[1] / dpi, img.shape[0] / dpi], frameon=frameon)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.show()


def plot_training_history(history):
    fig = plt.figure(figsize=(8, 10))
    plt.subplot(3, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['trn', 'val'], loc='lower right')

    plt.subplot(3, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['trn', 'val'], loc='upper right')
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.semilogy(history.history['loss'])
    plt.semilogy(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['trn', 'val'], loc='upper right')
    plt.tight_layout()


def evaluate_mlp(model, img_path="missing_pixels", out_file_path="images/mlp_output.png"):
    img = cv.imread(join(img_path, "balloon.bmp"), cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (256, 256), cv.INTER_CUBIC)
    img = img.astype(np.float64)

    inps = []
    tgts = []
    do_parse_img(img, inps, tgts)

    X_test = np.zeros((len(inps), NUM_INP_NODES))
    for idx in range(len(inps)):
        X_test[idx, :] = inps[idx]

    y_pred = model.predict(X_test)

    num_rows = img.shape[0]
    num_cols = img.shape[1]
    rec_img = np.zeros((num_rows, num_cols))

    blk_pos = 0
    for r0 in range(0, num_rows, BLOCK_SIZE):
        for c0 in range(0, num_cols, BLOCK_SIZE):
            for r in range(BLOCK_SIZE):
                for c in range(BLOCK_SIZE):
                    assert (r0 + r < num_rows)
                    assert (c0 + c < num_cols)
                    if not is_missing_pixel(r, c):
                        rec_img[r0 + r, c0 + c] = img[r0 + r, c0 + c]

            out = y_pred[blk_pos]
            blk_pos += 1

            pos = 0
            for r in range(BLOCK_SIZE):
                for c in range(BLOCK_SIZE):
                    assert (r0 + r < num_rows)
                    assert (c0 + c < num_cols)
                    if is_missing_pixel(r, c):
                        rec_img[r0 + r, c0 + c] = 255 * out[pos]
                        pos += 1

    cv.imwrite(out_file_path, rec_img)
    evaluate_from_image(out_file_path)


def evaluate_cnn(model, img_path="missing_pixels", out_file_path="images/cnn_output.png"):
    img = cv.imread(join(img_path, "balloon.bmp"), cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (256, 256), cv.INTER_LINEAR)
    img = img.astype(np.float64)

    X_test = []
    for i in range(0, img.shape[0], BLOCK_SIZE):
        for j in range(0, img.shape[1], BLOCK_SIZE):
            patch = img[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE]

            patch[3:5, 3:5] = 0.0
            X_test.append(patch)

    X_test = np.array(X_test) / 255.0
    X_test = X_test.reshape(-1, BLOCK_SIZE, BLOCK_SIZE, 1)

    y_pred = model.predict(X_test)

    num_rows = img.shape[0]
    num_cols = img.shape[1]
    rec_img = np.zeros((num_rows, num_cols))

    blk_pos = 0
    for r0 in range(0, num_rows, BLOCK_SIZE):
        for c0 in range(0, num_cols, BLOCK_SIZE):
            for r in range(BLOCK_SIZE):
                for c in range(BLOCK_SIZE):
                    assert (r0 + r < num_rows)
                    assert (c0 + c < num_cols)
                    if not is_missing_pixel(r, c):
                        rec_img[r0 + r, c0 + c] = img[r0 + r, c0 + c]

            out = y_pred[blk_pos]
            blk_pos += 1

            pos = 0
            for r in range(BLOCK_SIZE):
                for c in range(BLOCK_SIZE):
                    assert (r0 + r < num_rows)
                    assert (c0 + c < num_cols)
                    if is_missing_pixel(r, c):
                        rec_img[r0 + r, c0 + c] = 255 * out[pos]
                        pos += 1

    cv.imwrite(out_file_path, rec_img)
    evaluate_from_image(out_file_path)


def evaluate_from_image(predicted_image_path, img_path="missing_pixels"):
    img = cv.imread(join(img_path, "balloon.bmp"), cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (256, 256), cv.INTER_LINEAR)
    img = img.astype(np.float64)

    predicted_image = cv.imread(predicted_image_path, cv.IMREAD_GRAYSCALE)
    predicted_image = cv.resize(predicted_image, (256, 256), cv.INTER_LINEAR)
    predicted_image = predicted_image.astype(np.float64)

    num_rows = img.shape[0]
    num_cols = img.shape[1]
    rec_img = np.zeros((num_rows, num_cols))

    for r0 in range(0, num_rows, BLOCK_SIZE):
        for c0 in range(0, num_cols, BLOCK_SIZE):
            for r in range(BLOCK_SIZE):
                for c in range(BLOCK_SIZE):
                    assert (r0 + r < num_rows)
                    assert (c0 + c < num_cols)
                    if not is_missing_pixel(r, c):
                        rec_img[r0 + r, c0 + c] = img[r0 + r, c0 + c]

            for r in range(BLOCK_SIZE):
                for c in range(BLOCK_SIZE):
                    assert (r0 + r < num_rows)
                    assert (c0 + c < num_cols)
                    if is_missing_pixel(r, c):
                        rec_img[r0 + r, c0 + c] = predicted_image[r0 + r, c0 + c]

    err = rec_img - img
    print("MSE =", "{:.2f}".format((err ** 2).mean()))

    imshow(rec_img, "gray", dpi=72)
    plt.axis("off")