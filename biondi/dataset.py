import os
import glob
import re
import numpy as np
from skimage import io
import skimage
import skimage.morphology
import skimage.filters
import skimage.transform
import scipy
from scipy import ndimage
import openslide
import matplotlib.pyplot as plt
import pylab
import cv2 as cv
from tensorflow import keras
from sklearn.neighbors import KDTree
import h5py
import pandas as pd
import biondi.statistics
import pickle


def load_tif(tif_file):
    """
    Returns array of a .tif image.

    Keyword arguments:
    tif_file -- string; .tif filename
    """
    return io.imread(tif_file)


# Test to see if i can just directly return the ndimage.label function
#
def label_mask(mask):
    """
    Returns an array where each mask object has a unique label.

    Keyword arguments:
    mask -- an array of a binary mask
    """
    label, n = ndimage.label(mask)
    return label, n


def get_slices(label):
    """
    Returns slices to isolate and extract a labeled mask object.

    Keyword arguments:
    label -- a labeled mask array (usually output from label_mask())
    """
    x = ndimage.find_objects(label)
    return x


def get_padded_slices_v2(slices, pad, label):
    """
    Adds a buffer to object slices.

    Keyword arguments:
    slices -- list of slices for a single object (usually from get_slices())
    pad -- amount in pixel to pad/buffer the slices
    label -- a labeled mask array (usually output from label_mask())
    """
    new_slices = []
    for i, s in enumerate(slices):
        start = s.start
        stop = s.stop
        if i == 3:
            # code to prevent slicing of the channels
            new_start = 0
            new_stop = 3
            new_slices.append(slice(new_start, new_stop))
            continue

        if start - pad < 0:
            new_start = 0
        else:
            new_start = start - pad

        if stop + pad > label.shape[i]:
            new_stop = label.shape[i]
        else:
            new_stop = stop + pad

        if new_stop - new_start > 64:
            break
        else:
            new_slices.append(slice(new_start, new_stop))
    return tuple(new_slices)


def pad_to_64(cropped_shape, cropped_image):
    """
    Takes a cropped image and pads it to 64 pixel in each dimension.

    Keyword arguments:
    cropped_shape -- tuple; array shape of cropped_image
    cropped_image -- an array of an image
    """
    a, b, c, d, e, f = 0, 0, 0, 0, 0, 0
    for i in range(0, 3):
        if cropped_shape[i] % 2 == 1:
            if i == 0:
                a = int((64 - cropped_shape[i] + 1) / 2)
                b = a - 1

            elif i == 1:
                c = int((64 - cropped_shape[i] + 1) / 2)
                d = c - 1

            elif i == 2:
                e = int((64 - cropped_shape[i] + 1) / 2)
                f = e - 1

        else:
            if i == 0:
                a = int((64 - cropped_shape[i]) / 2)
                b = int((64 - cropped_shape[i]) / 2)

            elif i == 1:
                c = int((64 - cropped_shape[i]) / 2)
                d = int((64 - cropped_shape[i]) / 2)

            elif i == 2:
                e = int((64 - cropped_shape[i]) / 2)
                f = int((64 - cropped_shape[i]) / 2)
    padded_cropped_image = np.pad(cropped_image, ((a, b), (c, d), (e, f), (0, 0)), 'constant')
    return padded_cropped_image


def extract_padded_cropped_inclusions(mask_file, image_file, save_file, pad=5):
    """
    Extracts images of individual objects based on a binary mask.

    :param mask_file: filename of mask image
    :type mask_file: str
    :param image_file: filename of image
    :type image_file: str
    :param save_file: save filename
    :type save_file: str
    :param pad: amount of pixel padding when slicing out objects based on mask
    :type pad: int
    :return: an array of 3D padded cropped inclusions
    :rtype: ndarray
    """
    mask = load_tif(mask_file)
    original_image = load_tif(image_file)
    label, n = label_mask(mask)
    slices = get_slices(label)
    counter = 0
    images = []
    for i in range(n):
        new_slices = get_padded_slices_v2(slices[i], pad, label)
        if len(new_slices) < 4:
            continue
        else:
            cropped_image = original_image[new_slices]
            cropped_shape = cropped_image.shape
            images.append(pad_to_64(cropped_shape, cropped_image))
            counter += 1
    images = np.stack(images, axis=0)
    print('Number of extracted images:', len(images))
    np.save(save_file, images)
    return images


def randomize_and_segregate_dataset(images, labels):
    """
    Shuffles and splits images and their labels into 3 datasets:
    training, validation, and test datasets

    Keyword arguments:
    images -- an array of images
    labels -- an array of labels
    """
    rand_indices = np.random.choice(len(images), size=(np.rint((len(images) * 0.2)).astype('int64')), replace=False)
    test_batch = images[rand_indices]
    test_labels = labels[rand_indices]
    training_and_validation = np.delete(images, rand_indices, axis=0)
    training_and_validation_labels = np.delete(labels, rand_indices, axis=0)
    rand_indices2 = np.random.choice(len(training_and_validation),
                                     size=(np.rint((len(training_and_validation) * 0.2)).astype('int64')),
                                     replace=False)
    validation_batch = training_and_validation[rand_indices2]
    validation_labels = training_and_validation_labels[rand_indices2]
    training = np.delete(training_and_validation, rand_indices2, 0)
    traininglabels = np.delete(training_and_validation_labels, rand_indices2, 0)
    # below is new code to shuffle the training data
    rand_indices3 = np.random.choice(len(training), size=(len(training)), replace=False)
    training_batch = training[rand_indices3]
    training_labels = traininglabels[rand_indices3]
    return training_batch, training_labels, validation_batch, validation_labels, test_batch, test_labels


def load_wsi_coordinates(list_of_text_files):
    """
    Loads a list of WSI mapping coordinates data and returns a single array.

    Keyword arguments:
    list_of_text_files -- a list containing one or more .txt filesnames
    """
    files = []
    for i in list_of_text_files:
        x = (np.loadtxt(i)).astype(int)
        if len(x.shape) == 2:
            x = x[:, 1:3]
        elif len(x.shape) == 1:
            x.shape = (1, 3)
            x = x[:, 1:3]
        else:
            print('error')
            return
        files.append(x)
    combined_list = np.concatenate(files, axis=0)
    return combined_list


def load_wsi_coords_and_labels(coords_list, defined_label):
    """
    Loads a list of WSI mapping coordinates data and returns a single
    array of coordinates and an array of labels

    Keyword arguments:
    list_of_text_files -- a list containing one or more .txt filesnames
    defined_label -- a list of 3 binary label values ([0,0,0])
    """
    x = load_wsi_coordinates(coords_list)
    labels = np.zeros((len(x), 3), dtype=int)
    labels[:] = defined_label
    return x, labels


def wsi_cell_extraction_from_coords_v3(wsi_image_filename, im_size, coords, verbose=1):
    """
    Extracts cropped images at given coordinates from a larger image.

    Keyword arguments:
    wsi_image_filename -- string: filename of the image to extract from
    im_size -- size in pixels for extracted images
    coords -- list or array of coordinates in order
    """
    if type(wsi_image_filename) is str:
        wsi = openslide.open_slide(wsi_image_filename)
    else:
        wsi = wsi_image_filename
    cells = []
    counter = 0
    for i in coords:
        top_left_row = int(i[1] - (im_size / 2))
        top_left_column = int(i[0] - (im_size / 2))
        cells.append(wsi.read_region((top_left_column, top_left_row), 0, (im_size, im_size)))
        counter += 1
        if verbose == 1:
            print(counter, "out of", len(coords), "slices")
    cells = np.stack(cells, axis=0)
    cells = cells[:, :, :, :-1]
    return cells


def replace_w_b(x):
    """
    Replaces 'white' pixels with black.

    Keyword arguments:
    x -- an image array or array of images
    """
    counter = 0
    if len(x.shape) == 3:
        print('Single input image')
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                if (x[j, k, :] == np.array([255, 255, 255])).all():
                    x[j, k, :] = 0
    elif len(x.shape) == 4:
        print('Multiple input images')
        for i in range(len(x)):
            for j in range(x[i].shape[0]):
                for k in range(x[i].shape[1]):
                    if (x[i, j, k, :] == np.array([255, 255, 255])).all():
                        x[i, j, k, :] = 0
                        counter += 1
                        print(counter, i, 'out of', len(x))
    return x


def extract_im_and_labels_from_image(list_of_annotation_files, image_filename, im_size=64, from_mosaic=True):
    """
    Extracts cropped images and labels from a larger image (image mosaic
    or WSI) and annotation files.

    Keyword arguments:
    list_of_annotation_files -- list of strings; annotation filenames (.txt)
    image_filename -- string; filename of image mosaic or whole slide image
    im_size -- size in pixels of images to extract
    from_mosaic -- Boolean; whether source image is a mosaic or WSI; for
    determining whether to replace 'white' border pixels with black
    """
    counter = 0
    for i in list_of_annotation_files:
        if '_un.txt' in i:
            coords_u, labels_u = load_wsi_coords_and_labels([i], [0, 0, 0])
            counter += 1
        elif '_bi.txt' in i:
            coords_b, labels_b = load_wsi_coords_and_labels([i], [1, 0, 0])
            counter += 1
        elif '_v++.txt' in i:
            if '_bi_v++.txt' in i:
                coords_bvv, labels_bvv = load_wsi_coords_and_labels([i], [1, 0, 0])
                counter += 1
            else:
                coords_vv, labels_vv = load_wsi_coords_and_labels([i], [1, 0, 0])
                counter += 1
        elif '_mu.txt' in i:
            if '_bi_mu.txt' in i:
                coords_bm, labels_bm = load_wsi_coords_and_labels([i], [1, 1, 0])
                counter += 1
            else:
                coords_m, labels_m = load_wsi_coords_and_labels([i], [0, 1, 0])
                counter += 1
        elif '_v+.txt' in i:
            if '_bi_v+.txt' in i:
                coords_bv, labels_bv = load_wsi_coords_and_labels([i], [1, 0, 1])
                counter += 1
            elif '_mu_v+.txt' in i:
                if '_bi_mu_v+.txt' in i:
                    coords_bmv, labels_bmv = load_wsi_coords_and_labels([i], [1, 1, 1])
                    counter += 1
                else:
                    coords_mv, labels_mv = load_wsi_coords_and_labels([i], [0, 1, 1])
                    counter += 1
            else:
                coords_v, labels_v = load_wsi_coords_and_labels([i], [0, 0, 1])
                counter += 1
        elif '_vr.txt' in i:
            if '_bi_vr.txt' in i:
                coords_bvr, labels_bvr = load_wsi_coords_and_labels([i], [1, 0, 1])
                counter += 1
            elif '_mu_vr.txt' in i:
                if '_bi_mu_vr.txt' in i:
                    coords_bmvr, labels_bmvr = load_wsi_coords_and_labels([i], [1, 1, 1])
                    counter += 1
                else:
                    coords_mvr, labels_mvr = load_wsi_coords_and_labels([i], [0, 1, 1])
                    counter += 1
            else:
                coords_vr, labels_vr = load_wsi_coords_and_labels([i], [0, 0, 1])
                counter += 1
        else:
            print(i, 'is not properly named/recognized')

    if 'coords_bvv' or 'coords_vv' in locals():
        bvvc = []
        bvvl = []
        if 'coords_bvv' in locals():
            bvvc.append(coords_bvv)
            bvvl.append(labels_bvv)
        if 'coords_vv' in locals():
            bvvc.append(coords_vv)
            bvvl.append(labels_vv)
        if 'coords_b' in locals():
            bvvc.append(coords_b)
            bvvl.append(labels_b)
        if len(bvvc) != 0:
            coords_b = np.concatenate(bvvc, axis=0)
            labels_b = np.concatenate(bvvl, axis=0)
    if 'coords_vr' in locals():
        vc = []
        vl = []
        if 'coords_vr' in locals():
            vc.append(coords_vr)
            vl.append(labels_vr)
        if 'coords_v' in locals():
            vc.append(coords_v)
            vl.append(labels_v)
        if len(vc) != 0:
            coords_v = np.concatenate(vc, axis=0)
            labels_v = np.concatenate(vl, axis=0)
    if 'coords_bvr' in locals():
        bvc = []
        bvl = []
        if 'coords_bvr' in locals():
            bvc.append(coords_bvr)
            bvl.append(labels_bvr)
        if 'coords_bv' in locals():
            bvc.append(coords_bv)
            bvl.append(labels_bv)
        if len(bvc) != 0:
            coords_bv = np.concatenate(bvc, axis=0)
            labels_bv = np.concatenate(bvl, axis=0)
    if 'coords_mvr' in locals():
        mvc = []
        mvl = []
        if 'coords_mvr' in locals():
            mvc.append(coords_mvr)
            mvl.append(labels_mvr)
        if 'coords_mv' in locals():
            mvc.append(coords_mv)
            mvl.append(labels_mv)
        if len(mvc) != 0:
            coords_mv = np.concatenate(mvc, axis=0)
            labels_mv = np.concatenate(mvl, axis=0)
    if 'coords_bmvr' in locals():
        bmvc = []
        bmvl = []
        if 'coords_bmvr' in locals():
            bmvc.append(coords_bmvr)
            bmvl.append(labels_bmvr)
        if 'coords_bmv' in locals():
            bmvc.append(coords_bmv)
            bmvl.append(labels_bmv)
        if len(bmvc) != 0:
            coords_bmv = np.concatenate(bmvc, axis=0)
            labels_bmv = np.concatenate(bmvl, axis=0)

    if 'coords_u' not in locals():
        images_u = None
        labels_u = None
    else:
        images_u = wsi_cell_extraction_from_coords_v3(image_filename, im_size, coords_u)
        if from_mosaic:
            images_u = replace_w_b(images_u)
    if 'coords_b' not in locals():
        images_b = None
        labels_b = None
    else:
        images_b = wsi_cell_extraction_from_coords_v3(image_filename, im_size, coords_b)
        if from_mosaic:
            images_b = replace_w_b(images_b)
    if 'coords_m' not in locals():
        images_m = None
        labels_m = None
    else:
        images_m = wsi_cell_extraction_from_coords_v3(image_filename, im_size, coords_m)
        if from_mosaic:
            images_m = replace_w_b(images_m)
    if 'coords_v' not in locals():
        images_v = None
        labels_v = None
    else:
        images_v = wsi_cell_extraction_from_coords_v3(image_filename, im_size, coords_v)
        if from_mosaic:
            images_v = replace_w_b(images_v)
    if 'coords_bm' not in locals():
        images_bm = None
        labels_bm = None
    else:
        images_bm = wsi_cell_extraction_from_coords_v3(image_filename, im_size, coords_bm)
        if from_mosaic:
            images_bm = replace_w_b(images_bm)
    if 'coords_bv' not in locals():
        images_bv = None
        labels_bv = None
    else:
        images_bv = wsi_cell_extraction_from_coords_v3(image_filename, im_size, coords_bv)
        if from_mosaic:
            images_bv = replace_w_b(images_bv)
    if 'coords_mv' not in locals():
        images_mv = None
        labels_mv = None
    else:
        images_mv = wsi_cell_extraction_from_coords_v3(image_filename, im_size, coords_mv)
        if from_mosaic:
            images_mv = replace_w_b(images_mv)
    if 'coords_bmv' not in locals():
        images_bmv = None
        labels_bmv = None
    else:
        images_bmv = wsi_cell_extraction_from_coords_v3(image_filename, im_size, coords_bmv)
        if from_mosaic:
            images_bmv = replace_w_b(images_bmv)
    print('compare', len(list_of_annotation_files), counter)

    return [images_u, images_b, images_m, images_v, images_bm, images_bv, images_mv, images_bmv], [labels_u, labels_b,
                                                                                                   labels_m, labels_v,
                                                                                                   labels_bm, labels_bv,
                                                                                                   labels_mv,
                                                                                                   labels_bmv]


def merge_images_labels_without_none(im_var_list, label_var_list):
    """
    merges lists of images and labels together.

    Keyword arguments:
    im_var_list -- list of image arrays
    label_var_list -- list of label arrays
    """
    im = [i for i in im_var_list if i is not None]
    label = [i for i in label_var_list if i is not None]
    if len(im) != len(label):
        print("images and labels don't match")
        return
    # print(im)
    if len(im) == 0:
        print('No images to concatenate!')
        return
    images = np.concatenate(im, axis=0)
    labels = np.concatenate(label, axis=0)
    return images, labels


def images_and_full_annotations(list_of_annotations, source_image, im_size=64, from_mosaic=True):
    """
    Returns an array of images and an array of full morphology labels
    for a single image.

    Keyword arguments:
    list_of_annotations -- list of strings; annotation filenames (.txt)
    source_image -- string; filename of image mosaic or whole slide image
    im_size -- size in pixels of images to extract
    from_mosaic -- Boolean; whether source image is a mosaic or WSI; for
    determining whether to replace 'white' border pixels with black
    """
    ims, lbs = extract_im_and_labels_from_image(list_of_annotations, source_image, im_size=im_size,
                                                from_mosaic=from_mosaic)
    all_im, all_lb = merge_images_labels_without_none(ims, lbs)
    print('image dataset shape:', all_im.shape, 'label dataset shape:', all_lb.shape)
    return all_im, all_lb


def binary_annotation_from_full(full_annotations):
    """
    Returns an array of binary labels based on full annotations.

    Keyword arguments:
    full_annotations -- an array of full annotations
    """
    binary = []
    for i in range(len(full_annotations)):
        if full_annotations[i] != 0:
            binary.append(1)
        else:
            binary.append(0)
    binary = np.stack(binary, axis=0)
    return binary


def wsi_tile_extraction(wsi_filename, im_size, sample_name):
    """
    Converts a WSI in to an array of tiles and saves it as a numpy file.

    Keyword arguments:
    WSI_filename -- string; WSI filename
    im_size -- extracted tile size (in pixels)
    sample_name -- string; case number (ex. UCI-12-18 or A17-54)
    """
    wsi = openslide.open_slide(wsi_filename)
    dim = wsi.dimensions
    tiles = []
    # dim[0] = width, dim[1] = height
    im_num = (dim[1] // im_size) * (dim[0] // im_size)
    for i in range(dim[1] // im_size):
        for j in range(dim[0] // im_size):
            # j represent position on x-axis (different from usual which is row #)
            # i represent position on y-axis (different from usual which is column #)
            a = wsi.read_region((j * im_size, i * im_size), 0, (im_size, im_size))
            a = np.array(a)
            tiles.append(a[:, :, :-1])
            print(len(tiles), 'out of', im_num)
    tile_stack = np.stack(tiles, axis=0)
    print(tile_stack.shape)
    filename = sample_name + '_' + str(dim[1] // im_size) + 'rows_' + str(dim[0] // im_size) + 'columns.npy'
    np.save(filename, tile_stack)
    print('File saved as: ' + str(filename))


def wsi_tile_extraction_v2(wsi_filename, im_size, sample_name, half_res=False):
    """
    Converts a WSI in to an array of tiles and saves it as a numpy file.

    Keyword arguments:
    WSI_filename -- string; WSI filename
    im_size -- extracted tile size (in pixels) before halving resolution(if TRUE)
    sample_name -- string; case number (ex. UCI-12-18 or A17-54)
    half_res -- boolean; if true, halves the image resolution and size
    """
    wsi = openslide.open_slide(wsi_filename)
    dim = wsi.dimensions
    tiles = []
    # dim[0] = width, dim[1] = height
    im_num = (dim[1] // im_size) * (dim[0] // im_size)
    for i in range(dim[1] // im_size):
        for j in range(dim[0] // im_size):
            # j represent position on x-axis (different from usual which is row #)
            # i represent position on y-axis (different from usual which is column #)
            a = wsi.read_region((j * im_size, i * im_size), 0, (im_size, im_size))
            a = np.array(a)
            tiles.append(a[:, :, :-1])
            print(len(tiles), 'out of', im_num)
    tile_stack = np.stack(tiles, axis=0)
    if half_res:
        model = half_tile_resolution(im_size)
        tile_stack = model.predict(tile_stack)
        half_res_string = '_half_resolution'
    else:
        half_res_string = ''
    print(tile_stack.shape)
    if tile_stack.shape[1] != 1024:
        imsize = '_' + str(tile_stack.shape[1]) + 'imsize' + half_res_string
    else:
        imsize = ''
    filename = sample_name + imsize + '_' + str(dim[1] // im_size) + 'rows_' + str(dim[0] // im_size) + 'columns.npy'
    np.save(filename, tile_stack)
    print('File saved as: ' + str(filename))


def half_tile_resolution(im_size, channels=3):
    inputs = keras.Input(shape=(im_size, im_size, channels))
    out = keras.layers.AveragePooling2D()(inputs)
    model = keras.Model(inputs=inputs, outputs=out)
    return model


def generate_mosaic_template(filename):
    x = np.ones((10249, 10249, 3), dtype=np.uint8)
    x[:, :, :] = 255
    tiles = np.load(filename)
    rand_sample = np.random.choice(len(tiles), 500, replace=False)
    return x, tiles, rand_sample


def wsi_mosaic_start_stop_slices(counter):
    r = counter // 10
    c = counter % 10
    r_start = 1025 * r
    r_stop = 1025 * r + 1024
    c_start = 1025 * c
    c_stop = 1025 * c + 1024
    return r_start, r_stop, c_start, c_stop


def produce_mosaic_separated_rgb(filename, rgb=True):
    """
    Generates a random 10x10 mosaic from extracted WSI tiles along with
    metadata.

    Keyword arguments:
    filename -- string: .npy filename of 1024x1024 pixel WSI tiles
    rgb -- Boolean; whether to generate red, green, blue monochannel
    images in addition to the full rgb image
    """
    x, tiles, rand_sample = generate_mosaic_template(filename)
    y = []
    counter = 0
    for i in rand_sample:
        if np.amax(tiles[i, :, :, 2]) >= 85:
            r_start, r_stop, c_start, c_stop = wsi_mosaic_start_stop_slices(counter)
            x[r_start:r_stop, c_start:c_stop, :] = tiles[i]
            y.append(i)
            counter = counter + 1
            if counter >= 100:
                break
        else:
            continue
    plt.imshow(x[:, :, :])
    skimage.io.imsave(filename[:-21] + '_10x10_sample_mosaic.tif', x)
    np.save(filename[:-21] + '_10x10_sample_mosaic_metadata.npy', y)
    if rgb:
        skimage.io.imsave(filename[:-21] + '_10x10_sample_mosaic_red.tif', x[..., 0])
        skimage.io.imsave(filename[:-21] + '_10x10_sample_mosaic_green.tif', x[..., 1])
        skimage.io.imsave(filename[:-21] + '_10x10_sample_mosaic_blue.tif', x[..., 2])


def produce_additional_mosaic_separated_rgb(filename, rgb=True):
    """
    Generates an additional random 10x10 mosaic from extracted WSI tiles
    along with metadata.

    Keyword arguments:
    filename -- string: .npy filename of 1024x1024 pixel WSI tiles
    rgb -- Boolean; whether to generate red, green, blue monochannel
    images in addition to the full rgb image

    """
    case = re.search('(.+?)_(.+?)rows_(.+?)columns.npy', filename).group(1)
    previous_metadata = glob.glob(case + '*metadata*')
    prev_im = []
    for i in previous_metadata:
        prev_im.extend(np.load(i))

    x, tiles, rand_sample = generate_mosaic_template(filename)
    y = []
    counter = 0
    for i in rand_sample:
        if np.amax(tiles[i, :, :, 2]) >= 85 and i not in prev_im:
            r = counter // 10
            c = counter % 10
            r_start = 1025 * r
            r_stop = 1025 * r + 1024
            c_start = 1025 * c
            c_stop = 1025 * c + 1024
            x[r_start:r_stop, c_start:c_stop, :] = tiles[i]
            y.append(i)
            counter = counter + 1
            if counter >= 100:
                break
        elif np.amax(tiles[i, :, :, 2]) >= 85 and i in prev_im:
            print('found match...skipping', i)
        else:
            continue
    plt.imshow(x[:, :, :])
    skimage.io.imsave(case + '_10x10_sample_mosaic_' + str(len(previous_metadata) + 1) + '.tif', x)
    np.save(case + '_10x10_sample_mosaic_metadata_' + str(len(previous_metadata) + 1) + '.npy', y)
    if rgb:
        skimage.io.imsave(case + '_10x10_sample_mosaic_' + str(len(previous_metadata) + 1) + '_red.tif', x[..., 0])
        skimage.io.imsave(case + '_10x10_sample_mosaic_' + str(len(previous_metadata) + 1) + '_green.tif', x[..., 1])
        skimage.io.imsave(case + '_10x10_sample_mosaic_' + str(len(previous_metadata) + 1) + '_blue.tif', x[..., 2])


def data_augmentation(inputs, labels, iterations, dimensions, rotation=True, flip=True):
    """
    Generates augmented images by randomly flipping and rotating.

    Works for both 2D and 3D images

    Keyword arguments:
    inputs -- an array of images to augment
    labels -- an array of corresponding image labels
    iterations -- how many augmented images to generate per image
    dimensions -- 2 or 3; The number of image dimensions (do not
    include channels dimension)
    rotation -- Boolean; whether to randomly rotate images along each axis
    flip -- Boolean; whether to randomly flip images
    """
    augmented_data = []
    augmented_data_labels = []
    y = [(0, 1), (0, 2), (1, 2)]
    counter = 0
    for i in range(len(inputs)):
        for j in range(iterations):
            if flip:
                x = np.random.choice([0, 1, (0, 1), None])
                if x is not None:
                    im = np.flip(inputs[i], axis=x)
                else:
                    im = inputs[i]
            else:
                im = inputs[i]
            if rotation:
                if dimensions in (2, 3):
                    if dimensions == 2:
                        dimensions = 1
                    for k in range(dimensions):
                        im = scipy.ndimage.rotate(im, np.random.randint(360), axes=y[k], reshape=False)
                    else:
                        print('Provided number of dimensions is invalid! Should be either 2 or 3.')
                        return
            if not flip and rotation:
                print('What do you want me to do if no flipping or rotating?')
            augmented_data.append(im)
            augmented_data_labels.append(labels[i])
        counter += 1
        print('finished', counter, 'out of', len(inputs))
    augmented_data = np.stack(augmented_data, axis=0)
    augmented_data_labels = np.stack(augmented_data_labels, axis=0)
    return augmented_data, augmented_data_labels


def annotation_tool(name, filename, x):
    # May remove the follow if statement. Maybe not worth trying to recover a interrupted run (y).
    #
    #
    # Not sure if i should be checking in both locals and globals.
    if 'y' in locals() or 'y' in globals():
        answer1 = None
        while answer1 not in ("y", "n"):
            answer1 = input("Would you like rerun the annotation tool? This will delete unsaved annotaions! y(yes) or "
                            "n(no)?")
            if answer1 == "y":
                y = []
                j = 0
                break
            elif answer1 == "n":
                return
            else:
                print("Please enter y or n")
    # Check for existing in progress annotation file###
    save_list = glob.glob(name + '_annotations_*_out_of_*.npy')
    save_list_complete = glob.glob(name + '_annotations.npy')
    if save_list_complete:
        answer3 = None
        while answer3 not in ("y", "n"):
            answer3 = input("There is an existing completed annotation file! Would you like to continue? y(yes) or n("
                            "no)?")
            if answer3 == "y":
                y = []
                j = 0
            elif answer3 == "n":
                return
            else:
                print("Please enter y or n")
    elif save_list:
        answer2 = None
        while answer2 not in ("y", "n"):
            answer2 = input("Would you like to continue from save file! y(yes) or n(no)?")
            if answer2 == "y":
                y = (np.load(save_list[0]))
                y = y.tolist()
                j = len(y)
            elif answer2 == "n":
                y = []
                j = 0
            else:
                print("Please enter y or n")
    else:
        y = []
        j = 0
    while j < len(x):
        fig = pylab.figure()
        default_size = fig.get_size_inches()
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        for i in range(3):
            if i == 0:
                k = 0
            elif i == 1:
                k = 1
            else:
                k = 2
            z = x[j]
            z = z[:, :, :, ::-1]
            im = skimage.transform.resize(skimage.img_as_ubyte(skimage.exposure.equalize_adapthist(
                skimage.exposure.rescale_intensity(np.max(z, axis=k), in_range='uint12'))), (512, 512, 3), order=3)
            im[:, :, 2] = 0
            fig.add_subplot(1, 3, i + 1)
            pylab.imshow(im)
            pylab.axis('off')
        print('image #' + str(j + 1) + ' out of ' + str(len(x)))
        pylab.show()
        answer = None
        while answer not in ("y", "n", "u", "s", "ss"):
            answer = input("Is this an inclusion? y(yes), n(no), u(undo prev), s(save), ss(save and stop)")
            if answer == "y":
                y.append(1)
                j = j + 1
            elif answer == "n":
                y.append(0)
                j = j + 1
            elif answer == "u":
                del y[-1]
                j = j - 1
            elif answer == "s":
                save_list = glob.glob(name + '_annotations_*_out_of_*.npy')
                if j == 0:
                    print('Nothing to save!')
                elif save_list:
                    os.remove(save_list[0])
                d = np.asarray(y)
                np.save(name + '_annotations_' + str(j) + '_out_of_' + str(len(x)) + '.npy', d)
            elif answer == "ss":
                save_list = glob.glob(name + '_annotations_*_out_of_*.npy')
                if j == 0:
                    print('Nothing to save! Quiting...')
                    return
                elif save_list:
                    os.remove(save_list[0])
                d = np.asarray(y)
                np.save(name + '_annotations_' + str(j) + '_out_of_' + str(len(x)) + '.npy', d)
                return
            else:
                print("Please enter y, n, u, s, or ss.")
    p = np.asarray(y)
    d = np.asarray(y)
    d.shape = (len(x), 1)
    np.save(name + '_annotations.npy', d)
    save_list_remove = glob.glob(name + '_annotations_*_out_of_*.npy')
    if save_list_remove:
        os.remove(save_list_remove[0])
    print('You are DONE!!!')
    return d


def wsi_cell_extraction_from_tiles(wsi_tiles_filename, im_size, coords, save_filename):
    tiles = np.load(wsi_tiles_filename)
    cells = []
    counter = 0
    num_of_columns = int(re.search('rows_(.+?)columns.npy', wsi_tiles_filename).group(1))
    num_of_rows = int(re.search('tiles_(.+?)rows', wsi_tiles_filename).group(1))

    for i in range(len(tiles)):
        # get coords of image
        column = i % num_of_columns
        row = i // num_of_columns
        # pixel range
        c_start = im_size * column
        c_stop = c_start + im_size
        r_start = im_size * row
        r_stop = r_start + im_size
        for j in coords:
            if c_start <= j[0] <= c_stop and r_start <= j[1] <= r_stop:
                crop_r_start = int(round(j[1] - r_start) - 32)
                crop_r_stop = int(round(j[1] - r_start) + 32)
                crop_c_start = int(round(j[0] - c_start) - 32)
                crop_c_stop = int(round(j[0] - c_start) + 32)
                if crop_r_start < 0 or crop_c_start < 0 or crop_r_stop > 1024 or crop_c_stop > 1024:
                    counter = counter + 1
                    print(counter, "out of", len(coords), "slices", "not sliced", 'src_im', i)
                    continue
                else:
                    cells.append(tiles[i][crop_r_start:crop_r_stop, crop_c_start:crop_c_stop, :3])
                    counter = counter + 1
                    print(counter, "out of", len(coords), "slices", int(round(j[1] - r_start) - 32),
                          int(round(j[1] - r_start) + 32), int(round(j[0] - c_start) - 32),
                          int(round(j[0] - c_start) + 32), 'src_im', i)
            else:
                continue
    np.save(save_filename)
    return cells


def anc_params_from_tiles(tiles, coords, wsi_tiles_filename, bbox_size=64, tile_size=1024):
    anc_params = []
    num_of_columns = int(re.search('rows_(.+?)columns.npy', wsi_tiles_filename).group(1))
    print('Extracting bounding boxes.')
    for i in range(len(tiles)):
        tile_bbox_param = []
        # get coords of image
        column = i % num_of_columns
        row = i // num_of_columns
        # pixel range
        c_start = tile_size * column
        c_stop = c_start + tile_size
        r_start = tile_size * row
        r_stop = r_start + tile_size
        print(i + 1, 'out of', len(tiles), '---', c_start, c_stop, r_start, r_stop)

        for j in range(len(coords)):
            if c_start <= coords[j, 0] < c_stop and r_start <= coords[j, 1] < r_stop:
                x = int(bbox_size / 2)
                y0 = int(round(coords[j, 1] - r_start) - x)
                y1 = int(round(coords[j, 1] - r_start) + x)
                x0 = int(round(coords[j, 0] - c_start) - x)
                x1 = int(round(coords[j, 0] - c_start) + x)
                params = np.asarray([[y0, x0, y1, x1]])
                print('--', params, coords[j], (params < 1024).sum() + (0 <= params).sum() == 8)
                if (params < 1024).sum() + (0 <= params).sum() == 8:
                    tile_bbox_param.append(params)
        if len(tile_bbox_param) != 0:
            tile_bbox_param = np.concatenate(tile_bbox_param, axis=0)
            anc_params.append(tile_bbox_param)
        else:
            anc_params.append('None')
    return anc_params


def convert_anc_to_box(anc_params, boundingbox, cls_key='cls-c4', cls_mask_key='cls-c4-msk', reg_key='reg-c4',
                       reg_mask_key='reg-c4-msk'):
    """
    ***This code is meant for just 1 feature map*** Will not handle multiple feature maps.
    """
    cls = []
    reg = []
    cls_mask = []
    reg_mask = []
    print('Converting extracted anchors to box parameters.')
    for i in range(len(anc_params)):
        print(i + 1, 'out of', len(anc_params))
        box = boundingbox.convert_anc_to_box(anc_params[i], np.ones((anc_params[i].shape[0], 1)))
        cls.append(box[cls_key])
        cls_mask.append(box[cls_mask_key])
        reg.append(box[reg_key])
        reg_mask.append(box[reg_mask_key])
    cls = np.array(cls)
    cls.shape = (len(cls),
                 boundingbox.params['anchor_gsizes'][0][0],
                 boundingbox.params['anchor_gsizes'][0][0],
                 boundingbox.params['classes'])
    cls_mask = np.array(cls_mask)
    cls_mask.shape = (len(cls_mask),
                      boundingbox.params['anchor_gsizes'][0][0],
                      boundingbox.params['anchor_gsizes'][0][0],
                      boundingbox.params['classes'])
    reg = np.array(reg)
    reg.shape = (len(reg),
                 boundingbox.params['anchor_gsizes'][0][0],
                 boundingbox.params['anchor_gsizes'][0][0],
                 4)
    reg_mask = np.array(reg_mask)
    reg_mask.shape = (len(reg_mask),
                      boundingbox.params['anchor_gsizes'][0][0],
                      boundingbox.params['anchor_gsizes'][0][0],
                      4)
    return cls, reg, cls_mask, reg_mask


def per_sample_tile_normalization(sorted_tiles, per_channel=False):
    """Per sample tile normalization. Channels are normalized individually."""
    if per_channel:
        images = []
        for i in range(len(sorted_tiles)):
            # print(i + 1, 'out of', len(sorted_tiles))
            sample = sorted_tiles[i]
            image = (sample - np.mean(sample, axis=tuple(range(sample.ndim - 1)))) / np.std(sample, axis=tuple(
                range(sample.ndim - 1)))
            images.append(image)
        images = np.array(images)
        return images
    else:
        images = []
        for i in range(len(sorted_tiles)):
            # print(i + 1, 'out of', len(sorted_tiles))
            sample = sorted_tiles[i]
            image = (sample - np.mean(sample)) / np.std(sample)
            images.append(image)
        images = np.array(images)
        return images


def normalized_tiles_and_bbox_params_from_wsi_tiles(wsi_tiles_filename, coords, boundingbox, normalize=False,
                                                    bbox_size=64, tile_size=1024, per_channel=False):
    tiles = np.load(wsi_tiles_filename)
    if tiles.shape[3] == 4:
        tiles = tiles[:, :, :, :-1]
    anc = anc_params_from_tiles(tiles=tiles, coords=coords, wsi_tiles_filename=wsi_tiles_filename, bbox_size=bbox_size,
                                tile_size=tile_size)
    sorted_tiles = []
    sorted_anc = []
    print('Removing tiles without bounding boxes.')
    for i in range(len(anc)):
        print(i + 1, 'out of', len(anc))
        if anc[i] != 'None':
            sorted_tiles.append(tiles[i])
            sorted_anc.append((anc[i]))
    sorted_tiles = np.array(sorted_tiles)
    tiles = None
    cls, reg, cls_mask, reg_mask = convert_anc_to_box(sorted_anc, boundingbox)
    if normalize:
        images = per_sample_tile_normalization(sorted_tiles, per_channel=per_channel)
    else:
        images = sorted_tiles
    return images, cls, reg, cls_mask, reg_mask


def anc_params_from_mosaics(mosaic_metadata, coords, bbox_size=64, tile_size=1024):
    # TODO: check if this function is used and implement use of tile_size
    anc_params = []
    for i in range(len(mosaic_metadata)):
        tile_bbox_param = []
        r_start, r_stop, c_start, c_stop = wsi_mosaic_start_stop_slices(i)
        for j in range(len(coords)):
            if c_start <= coords[j, 0] < c_stop and r_start <= coords[j, 1] < r_stop:
                x = int(bbox_size / 2)
                y0 = int(round(coords[j, 1] - r_start) - x)
                y1 = int(round(coords[j, 1] - r_start) + x)
                x0 = int(round(coords[j, 0] - c_start) - x)
                x1 = int(round(coords[j, 0] - c_start) + x)
                params = np.asarray([[y0, x0, y1, x1]])
                print('--', params, coords[j], (params < 1024).sum() + (0 <= params).sum() == 8)
                if (params < 1024).sum() + (0 <= params).sum() == 8:
                    tile_bbox_param.append(params)
        if len(tile_bbox_param) != 0:
            tile_bbox_param = np.concatenate(tile_bbox_param, axis=0)
            anc_params.append(tile_bbox_param)
        else:
            anc_params.append('None')
    return anc_params


def normalized_tiles_and_bbox_params_from_wsi_mosaic(wsi_tiles_filename, mosaic_metadata, coords, boundingbox,
                                                     normalize=False,
                                                     bbox_size=64, tile_size=1024, per_channel=False):
    mosaic = np.load(mosaic_metadata)
    tiles = np.load(wsi_tiles_filename)
    anc = anc_params_from_mosaics(mosaic_metadata=mosaic, coords=coords, bbox_size=bbox_size, tile_size=tile_size)
    sorted_tiles = []
    sorted_anc = []
    print('Removing tiles without bounding boxes.')
    for i in range(len(anc)):
        print(i + 1, 'out of', len(anc))
        if anc[i] != 'None':
            sorted_tiles.append(tiles[mosaic[i]])
            sorted_anc.append((anc[i]))
    sorted_tiles = np.array(sorted_tiles)
    cls, reg, cls_mask, reg_mask = convert_anc_to_box(sorted_anc, boundingbox)
    if normalize:
        images = per_sample_tile_normalization(sorted_tiles, per_channel=per_channel)
    else:
        images = sorted_tiles
    return images, cls, reg, cls_mask, reg_mask


def convert_anc_to_box_v2_2d(anc_params, boundingbox):
    """
    ***This code is meant for multiple feature maps***
    """
    boxes = {}
    for h in boundingbox.params['inputs_shapes'].keys():
        boxes[h] = []
    print('Converting extracted anchors to box parameters.')
    for i in range(len(anc_params)):
        print(i + 1, 'out of', len(anc_params))
        box = boundingbox.convert_anc_to_box(anc_params[i], np.ones((anc_params[i].shape[0], 1)))
        for j in box.keys():
            boxes[j].append(box[j])
    for k in boxes.keys():
        # this will provide an array with 4 dims for a 2d image, will not work for 3d!
        boxes[k] = np.concatenate(boxes[k], axis=0)
        print(k, boxes[k].shape)
    return boxes


def normalized_tiles_and_bbox_params_from_wsi_mosaic_v2(wsi_tiles_filename, mosaic_metadata, coords, boundingbox,
                                                        normalize=False, bbox_size=64, tile_size=1024,
                                                        per_channel=False):
    mosaic = np.load(mosaic_metadata)
    tiles = np.load(wsi_tiles_filename)
    if tiles.shape[3] == 4:
        tiles = tiles[:, :, :, :-1]
    anc = anc_params_from_mosaics(mosaic_metadata=mosaic, coords=coords, bbox_size=bbox_size, tile_size=tile_size)
    sorted_tiles = []
    sorted_anc = []
    print('Removing tiles without bounding boxes.')
    for i in range(len(anc)):
        print(i + 1, 'out of', len(anc))
        if anc[i] != 'None':
            sorted_tiles.append(tiles[mosaic[i]])
            sorted_anc.append((anc[i]))
    sorted_tiles = np.array(sorted_tiles)
    del tiles
    im_boxes = convert_anc_to_box_v2_2d(sorted_anc, boundingbox)
    if normalize:
        images = per_sample_tile_normalization(sorted_tiles, per_channel=per_channel)
    else:
        images = sorted_tiles
    return images, im_boxes


def normalized_tiles_and_bbox_params_from_wsi_tiles_v2(wsi_tiles_filename, coords, boundingbox, normalize=False,
                                                       bbox_size=64, tile_size=1024, per_channel=False):
    tiles = np.load(wsi_tiles_filename)
    if tiles.shape[3] == 4:
        tiles = tiles[:, :, :, :-1]
    anc = anc_params_from_tiles(tiles=tiles, coords=coords, wsi_tiles_filename=wsi_tiles_filename, bbox_size=bbox_size,
                                tile_size=tile_size)
    sorted_tiles = []
    sorted_anc = []
    print('Removing tiles without bounding boxes.')
    for i in range(len(anc)):
        print(i + 1, 'out of', len(anc))
        if anc[i] != 'None':
            sorted_tiles.append(tiles[i])
            sorted_anc.append((anc[i]))
    sorted_tiles = np.array(sorted_tiles)
    del tiles
    im_boxes = convert_anc_to_box_v2_2d(sorted_anc, boundingbox)
    if normalize:
        images = per_sample_tile_normalization(sorted_tiles, per_channel=per_channel)
    else:
        images = sorted_tiles
    return images, im_boxes


# code to extract nuclei from wsi
def threshold_label_segment(wsi_tile):
    threshold = skimage.filters.threshold_isodata(wsi_tile[:, :, 2])
    binary_mask = wsi_tile[:, :, 2] > threshold
    binary_mask = ndimage.morphology.binary_fill_holes(skimage.morphology.dilation(
        skimage.morphology.dilation(skimage.morphology.erosion(skimage.morphology.erosion(binary_mask)))))
    binary_mask = np.uint8(binary_mask)

    # sure background
    sure_bg = np.uint8(
        skimage.morphology.dilation(skimage.morphology.dilation(skimage.morphology.dilation(binary_mask))))

    # sure foreground
    dist_transform = ndimage.distance_transform_edt(binary_mask)
    sure_fg_threshold = skimage.filters.threshold_isodata(dist_transform)
    sure_fg = np.uint8(dist_transform > sure_fg_threshold)

    # unknown region
    unknown = np.subtract(sure_bg, sure_fg)

    # label mask
    labels, _ = ndimage.label(sure_fg)
    labels = labels + 1
    labels[unknown == 1] = 0

    # add empty channels for watershed function
    binary_mask_3c = np.stack([binary_mask, np.zeros_like(binary_mask), np.zeros_like(binary_mask)], axis=2)

    # watershed segmented mask
    segmented_mask = cv.watershed(binary_mask_3c, labels)

    return segmented_mask, binary_mask


def get_wsi_coords_of_cells_from_tiles(i, coords, wsi_tiles_filename, tile_size):
    num_of_columns = int(re.search('rows_(.+?)columns.npy', wsi_tiles_filename).group(1))
    column = i % num_of_columns
    row = i // num_of_columns
    c_start = tile_size * column
    r_start = tile_size * row
    wsi_coords = []
    for j in coords:
        # TODO: revision comment: looks like code is treating coords as yx instead of xy,
        #  could be breaking other code that uses this
        # TODO: need to correct this and other code to use xy coords
        wsi_coords.append([r_start + j[0], c_start + j[1]])
    if len(wsi_coords) == 0:
        return None
    else:
        return np.array(wsi_coords)


def extract_cells_using_mask(wsi_tiles_filename, get_wsi_ccords=False, tile_size=1024):
    # function to get wsi coords is providing yx instead xy
    # TODO: make sure this code works once WSI_coords funciton is fixed
    wsi_tiles = np.load(wsi_tiles_filename)
    cells = []
    if get_wsi_ccords:
        wsi_coords = []
    for i in range(len(wsi_tiles)):
        if np.sum(wsi_tiles[i, :, :, 2]) == 0:
            continue
        segmented_mask, binary_mask = threshold_label_segment(wsi_tiles[i])
        # com_coords provides yx coords, not xy
        # TODO: edit com_coords so they are in xy format
        com_coords = ndimage.center_of_mass(binary_mask, segmented_mask, range(2, np.max(segmented_mask) + 1))
        com_coords = np.array(com_coords)
        com_coords = com_coords[~np.any(np.isnan(com_coords), axis=1)].astype(np.int32)
        delete1 = np.nonzero(com_coords - 32 < 0)[0]
        delete2 = np.nonzero(com_coords + 32 >= tile_size)[0]
        delete = np.concatenate([delete1, delete2])
        trimmed_com_coords = np.delete(com_coords, delete, axis=0)
        for j in trimmed_com_coords:
            cells.append(wsi_tiles[i, j[0] - 32:j[0] + 32, j[1] - 32:j[1] + 32, :])
        if get_wsi_ccords:
            wsi_coords.append(get_wsi_coords_of_cells_from_tiles(i, trimmed_com_coords, wsi_tiles_filename, tile_size))
        print(i + 1, 'out of', len(wsi_tiles))  # might replace with progress bar
    if get_wsi_ccords:
        cells = np.stack(cells)
        wsi_coords = [x for x in wsi_coords if x is not None]
        wsi_coords = np.concatenate(wsi_coords, axis=0)
        if len(wsi_coords) != len(cells):
            print('Number of extracted cells and coordinates do not match!')
        return cells, wsi_coords[:, ::-1]
    else:
        return np.array(cells)


def annotate_threshold_coords_using_manual_coords(thresh_coords, manual_coords, radius=20):
    mt = KDTree(manual_coords)
    nn = mt.query_radius(thresh_coords, radius, count_only=True)
    annotations = np.where(nn > 0, 1, 0)
    return annotations


def local_to_global_coords_retinanet(local_coords, wsi_tiles_filename, tile_size):
    # local_coords is a list of arrays which should be in xy format
    num_of_columns = int(re.search('rows_(.+?)columns.npy', wsi_tiles_filename).group(1))
    global_coords = []
    for i in range(len(local_coords)):
        if len(local_coords[i]) == 0:
            continue
        # get coords of tile
        column = i % num_of_columns
        row = i // num_of_columns
        # pixel range
        c_start = tile_size * column
        r_start = tile_size * row
        for j in range(len(local_coords[i])):
            global_coords.append([c_start + local_coords[i][j, 0], r_start + local_coords[i][j, 1]])
    if len(global_coords) == 0:
        return None
    else:
        return np.array(global_coords)


def convert_mosaic_coords_to_wsi(coords, metadata, wsi_tiles_filename, tile_size=1024):
    """

    :param coords:
    :type coords:
    :param metadata:
    :type metadata:
    :param wsi_tiles_filename:
    :type wsi_tiles_filename:
    :param tile_size:
    :type tile_size:
    :return:
    :rtype:
    """
    num_of_columns = int(re.search('rows_(.+?)columns.npy', wsi_tiles_filename).group(1))
    # coords must be in xy format
    grid_coords = coords // 1025
    tile_coords = coords - (grid_coords * 1025)
    tile_index = grid_coords[:, 1] * 10 + grid_coords[:, 0]
    for i in range(len(tile_index)):
        tile_index[i] = metadata[tile_index[i]]  # mosaic tile_index now represents wsi tile index
    columns = tile_index % num_of_columns  # (n,) array
    rows = tile_index // num_of_columns  # (n,) array
    c_start = tile_size * columns  # (n,) array
    r_start = tile_size * rows  # (n,) array
    tile_start_coords = np.stack([c_start, r_start], axis=1)  # tile_start_coords are top left coords for each tile
    if tile_coords.shape == tile_start_coords.shape:
        return tile_coords + tile_start_coords
    else:
        print('arrays do not match')


def get_retinanet_training_dictionary_from_mosaic(wsi_tiles_filename, coords, boundingbox, normalize=False,
                                                  bbox_size=64, tile_size=1024):
    tiles, boxes = normalized_tiles_and_bbox_params_from_wsi_tiles_v2(
        wsi_tiles_filename,
        coords,
        boundingbox,
        normalize=normalize,
        bbox_size=bbox_size,
        tile_size=tile_size,
    )
    boxes['dat'] = tiles
    for key in boxes.keys():
        boxes[key] = np.expand_dims(boxes[key], axis=1)
    return boxes


def randomize_and_segregate_dataset_retinanet_dictionary(dictionary, validation_percent=0.15, test_percent=0.15):
    """
    Shuffles and splits images and box anchors into 3 dataset dictionaries:
    training, validation, and test datasets

    Keyword arguments:
    images -- an array of images
    labels -- an array of labels
    """

    p = np.random.permutation(len(dictionary['dat']))
    test_number = int(len(dictionary['dat']) * test_percent)
    validation_number = int(len(dictionary['dat']) * validation_percent)
    test_indices = p[0:test_number]
    validation_indices = p[test_number:(test_number + validation_number)]
    training_indices = p[(test_number + validation_number):]
    test_dic = {key: dictionary[key][test_indices] for key in dictionary.keys()}
    validation_dic = {key: dictionary[key][validation_indices] for key in dictionary.keys()}
    training_dic = {key: dictionary[key][training_indices] for key in dictionary.keys()}
    return training_dic, validation_dic, test_dic


def retinanet_generator(data, batchsize=1, normalize=True, per_channel=False, two_channel=False):
    i = 0
    keys = data.keys()
    while True:
        if i == (len(data['dat']) // batchsize):
            i = 0
            p = np.random.permutation(len(data['dat']))
            for key in keys:
                data[key] = data[key][p]
        start = i * batchsize
        stop = start + batchsize
        xbatch = {}
        ybatch = {}
        for key in keys:
            if 'dat' in key:
                if normalize:
                    if two_channel:
                        xbatch[key] = per_sample_tile_normalization(data[key][start:stop, ..., 1:3],
                                                                    per_channel=per_channel)
                    else:
                        xbatch[key] = per_sample_tile_normalization(data[key][start:stop], per_channel=per_channel)
                else:
                    if two_channel:
                        xbatch[key] = data[key][start:stop, ..., 1:3]
                    else:
                        xbatch[key] = data[key][start:stop]
            elif 'msk' in key:
                xbatch[key] = data[key][start:stop]
            else:
                ybatch[key] = data[key][start:stop]
        i += 1
        yield xbatch, ybatch


def retinanet_eval_generator(data, batchsize=1, normalize=True, per_channel=False):
    full_steps = (len(data['dat']) // batchsize)
    if len(data['dat']) % batchsize != 0:
        partial_step = 1
    else:
        partial_step = 0
    keys = data.keys()
    for i in range(full_steps + partial_step):
        start = i * batchsize
        stop = start + batchsize
        if stop > len(data['dat']):
            stop = len(data['dat'])
        xbatch = {}
        ybatch = {}
        for key in keys:
            if 'dat' in key:
                if normalize:
                    xbatch[key] = per_sample_tile_normalization(data[key][start:stop], per_channel=per_channel)
                else:
                    xbatch[key] = data[key][start:stop]
            elif 'msk' in key:
                xbatch[key] = data[key][start:stop]
            else:
                ybatch[key] = data[key][start:stop]
        i += 1
        yield xbatch, ybatch


def retinanet_evaluation(evaluation_generator, model, bb):
    ious = {
        'med': [],
        'p25': [],
        'p75': [],
    }
    for x, y in evaluation_generator:
        box = model.predict(x)
        # list check taken from peter's tutorial. Not sure if needed for my code but included it just in case.
        if type(box) is list:
            box = {name: pred for name, pred in zip(model.output_names, box)}
        anchors_pred, _ = bb.convert_box_to_anc(box)
        anchors_true, _ = bb.convert_box_to_anc(y)

        curr = []
        for pred, true in zip(anchors_pred, anchors_true):
            for p in pred:
                iou = bb.calculate_ious(box=p, anchors=true)
                if iou.size > 0:
                    curr.append(np.max(iou))
                else:
                    curr.append(0)
        if len(curr) == 0:
            curr = [0]
        ious['med'].append(np.median(curr))
        ious['p25'].append(np.percentile(curr, 25))
        ious['p75'].append(np.percentile(curr, 75))
    ious = {k: np.array(v) for k, v in ious.items()}

    # --- Define columns
    df = pd.DataFrame(index=np.arange(ious['med'].size))
    df['iou_median'] = ious['med']
    df['iou_p-25th'] = ious['p25']
    df['iou_p-75th'] = ious['p75']

    # --- Print accuracy
    print(df['iou_median'].median())
    print(df['iou_p-25th'].median())
    print(df['iou_p-75th'].median())
    return df


def retinanet_prediction_generator(images, boundingbox, per_channel=False):
    # TODO: consider changing name since this is not a generator or modify code to make it a generator
    pred_dic = {'dat': per_sample_tile_normalization(np.expand_dims(images, axis=1), per_channel=per_channel)}
    for key in boundingbox.params['inputs_shapes'].keys():
        if 'msk' in key:
            pred_dic[key] = np.zeros(shape=(len(images),) + tuple(boundingbox.params['inputs_shapes'][key]))
    return pred_dic


def retinanet_validation_generator(validation_dict, per_channel=False):
    # TODO: does not work as val_gen for model training, consider removing or recoding
    val_dict = validation_dict.copy()
    val_dict['dat'] = per_sample_tile_normalization(validation_dict['dat'], per_channel=per_channel)
    return val_dict


def retinanet_prediction(images, model, boundingbox):
    # want to add path boolean and code load model if model is a filepath
    output = model.predict(retinanet_prediction_generator(images, boundingbox))
    output_dic = {name: pred for name, pred in zip(model.output_names, output)}
    return output_dic


def cpec_coords_from_anc(anc, wsi_tiles_filename, tile_size, half_res=False):
    # anc is a list(2: anchor coords and classes) of a list of arrays (1 array per image tile)
    # anc arrays are local coordinates for the tile
    # convert tile coords to WSI coords
    local_coords = []
    for j in range(len(anc[0])):
        if len(anc[0][j]) != 0:
            y_coords = (anc[0][j][:, 2] + anc[0][j][:, 0]) / 2
            x_coords = (anc[0][j][:, 3] + anc[0][j][:, 1]) / 2
            local_coords.append(np.stack([x_coords, y_coords], axis=1))
        else:
            local_coords.append(anc[0][j])
    wsi_coords = local_to_global_coords_retinanet(local_coords, wsi_tiles_filename, tile_size)
    if not half_res:
        return wsi_coords
    elif half_res:
        return (wsi_coords * 2).astype(int)
    else:
        print('half_res parameter must be a boolean!')
        return


def tile_sample_hdf5_generator(tile_stack_filename, sample_size=100, name=None):
    # TODO: modify to allow re functions to work when not within the file folder or when providing a full file path
    # TODO: add naming override functionality
    num_of_columns = int(re.search('rows_(.+?)columns.npy', tile_stack_filename).group(1))
    num_of_rows = int(re.search('_(.{1,4}?)rows', tile_stack_filename).group(1))
    if name:
        wsi_name = name
    else:
        wsi_name = re.search('(^.{5,13}?)_', tile_stack_filename).group(1)
    filename = wsi_name + '_tile_sample_'
    previous_metadata = glob.glob(filename + '*.hdf5')
    max_previous = max([int(re.search('sample_(.{1,2}?).hdf5', i).group(1)) for i in previous_metadata] + [0])
    full_filename = filename + str(max_previous + 1) + '.hdf5'
    tile_stack = np.load(tile_stack_filename)
    if max_previous == 0:
        p = np.random.permutation(len(tile_stack))
        delete_p = []
        for i in range(len(p)):
            if np.amax(tile_stack[p[i], :, :, 2]) < 85:
                delete_p.append(i)
        p = np.delete(p, delete_p, axis=0)
    else:
        f_old = h5py.File(filename + '1.hdf5', 'r')
        p = f_old['full_randomized_tile_indices']
    print('Saving...')
    f = h5py.File(full_filename, 'w')
    dset1 = f.create_dataset('images', data=tile_stack[p[:sample_size * (max_previous + 1)]])
    dset2 = f.create_dataset('rows-columns', data=np.array([num_of_rows, num_of_columns]))
    dset3 = f.create_dataset('tile_index', data=p[:sample_size * (max_previous + 1)])
    dset4 = f.create_dataset('full_randomized_tile_indices', data=p)
    print('Saved as:', full_filename)
    return


def tile_sample_hdf5_generator_v2(wsi_filename, im_size=1024, sample_size=100, name=None):
    if name:
        wsi_name = name
    else:
        wsi_name = re.search('(^.*?) ', wsi_filename).group(1)
    filename = wsi_name + '_tile_sample_'
    previous_metadata = glob.glob(filename + '*.hdf5')
    max_previous = max([int(re.search('sample_(.{1,2}?).hdf5', i).group(1)) for i in previous_metadata] + [0])
    full_filename = filename + str(max_previous + 1) + '.hdf5'
    wsi = openslide.open_slide(wsi_filename)
    dim = wsi.dimensions
    grid_height = dim[1] // im_size
    grid_width = dim[0] // im_size
    im_num = grid_height * grid_width
    if max_previous == 0:
        p = np.random.permutation(im_num)
    else:
        f_old = h5py.File(filename + '1.hdf5', 'r')
        p = f_old['full_randomized_tile_indices']
    tile_stack = []
    for index in p[:sample_size * (max_previous + 1)]:
        # determine row in WSI
        i = index // grid_width
        # determine column in WSI
        j = index % grid_width
        a = wsi.read_region((j * im_size, i * im_size), 0, (im_size, im_size))
        tile_stack.append(np.array(a)[:, :, :-1])
    f = h5py.File(full_filename, 'w')
    dset1 = f.create_dataset('images', data=np.stack(tile_stack, axis=0))
    dset2 = f.create_dataset('rows-columns', data=np.array([grid_height, grid_width]))
    dset3 = f.create_dataset('tile_index', data=p[:sample_size * (max_previous + 1)])
    dset4 = f.create_dataset('full_randomized_tile_indices', data=p)
    print('Saved as:', full_filename)


def unet_generator(imgs, masks, per_channel=False):
    i = 0
    names = list(imgs.keys())
    while True:
        if i == len(names):
            i = 0
            p = np.random.permutation(len(names))
            names = names[p]
        img = per_sample_tile_normalization(np.expand_dims(imgs[names[i]][:], axis=0), per_channel=per_channel)
        msk = np.expand_dims(masks[names[i]][:], axis=0)
        i += 1
        yield img, msk


def sliding_window_generator(img, batchsize=16, window_size=64):
    # img should be a 4D array
    # window_size must be even
    windows = skimage.util.view_as_windows(np.pad(img,
                                                  ((int(window_size / 2), int(window_size / 2) - 1),
                                                   (int(window_size / 2), int(window_size / 2) - 1),
                                                   (int(window_size / 2), int(window_size / 2) - 1),
                                                   (0, 0)),
                                                  'constant',
                                                  constant_values=0),
                                           (window_size, window_size, window_size, 3),
                                           step=1)
    counter = 0
    im_batch = []
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            for k in range(windows.shape[2]):
                # should I replace 0, in slice with :?
                im_batch.append(windows[i, j, k, 0, ...])
                counter += 1
                if counter == batchsize:
                    yield np.stack(im_batch)
                    counter = 0
                    im_batch = []


def count_num_objs(msk, threshold, display_im=True):
    import copy
    labeled_msk = skimage.measure.label(msk, return_num=True, connectivity=3)
    # print('Before cleanup:', labeled_msk[1], 'objects')
    obj_vol = np.unique(labeled_msk[0], return_counts=True)
    # Number of object at each found voxel size
    vol_prevalence = np.unique(obj_vol[1], return_counts=True)
    obj_to_drop = obj_vol[0][obj_vol[1] < threshold]
    # print('After cleanup:', labeled_msk[1] - len(obj_to_drop), 'objects')
    msk2 = copy.deepcopy(labeled_msk[0])
    for label in obj_to_drop:
        msk2[msk2 == label] = 0
    msk2[msk2 > 0] = 1
    if display_im:
        # before cleanup
        print('Before cleanup:', labeled_msk[1], 'objects')
        plt.imshow(np.max(msk, axis=0))
        plt.show()
        # after cleanup
        print('After cleanup:', labeled_msk[1] - len(obj_to_drop), 'objects')
        plt.imshow(np.max(msk2, axis=0))
        plt.show()
    return msk2


def generate_3d_binary_mask(image_fn_list, channel_index=1, voxel_threshold=50, sigma=1, kernel=(5, 7, 7),
                            save_tiff=False, save_npy=True, src_fldr=None, dst_fldr=None):
    """
    *** Requires dev version of scipy for background subtraction ***
    Generates binary masked based on a single channel. Applies gaussian blurring followed by background subtraction
    using the rolling ball algorithm (similar to imagej) and finally generates a binary image based on ostu
    thresholding.
    :param sigma:
    :type sigma:
    :param kernel:
    :type kernel:
    :param dst_fldr:
    :type dst_fldr:
    :param src_fldr:
    :type src_fldr:
    :param save_npy:
    :type save_npy:
    :param save_tiff:
    :type save_tiff:
    :param image_fn_list:
    :type image_fn_list:
    :param channel_index:
    :type channel_index:
    :param voxel_threshold:
    :type voxel_threshold:
    :return:
    :rtype:
    """
    from skimage import filters, restoration

    def check_input_number(image_filename):
        while True:
            try:
                user_input = int(input(f'{image_filename} has >3 channels! Specify the desired channel index.'))
            except ValueError:
                print("Not an integer! Try again.")
                continue
            else:
                return user_input

    def create_mask(image_filename, source_folder=None, channel_index_=None, voxel_threshold_=None,
                    sigma_=None, kernel_=None):
        if source_folder:
            image = skimage.io.imread(source_folder + image_filename)
        else:
            image = skimage.io.imread(image_filename)
        if image.shape[-1] != 3:
            blurred_image = filters.gaussian(image[..., check_input_number(image_filename)], sigma=sigma_,
                                             preserve_range=True)
        else:
            blurred_image = filters.gaussian(image[..., channel_index_], sigma=sigma_, preserve_range=True)
        background = restoration.rolling_ball(blurred_image, kernel=restoration.ellipsoid_kernel(kernel_, 0.1))
        bkgrd_sub = blurred_image - background
        ostu_thresh = filters.threshold_otsu(bkgrd_sub)
        binary_msk = bkgrd_sub >= ostu_thresh
        cleanedup = count_num_objs(binary_msk, threshold=voxel_threshold_, display_im=False)
        return cleanedup

    for image_fn in image_fn_list:
        print(f'Working on {image_fn}', end=' ')
        mask = create_mask(image_fn, source_folder=src_fldr, channel_index_=channel_index,
                           voxel_threshold_=voxel_threshold, sigma_=sigma, kernel_=kernel)
        if save_npy:
            if dst_fldr:
                np.save(dst_fldr + image_fn[:-4] + '_mask.npy', mask)
            else:
                np.save(image_fn[:-4] + '_mask.npy', mask)
        if save_tiff:
            mask[mask == 1] = 255
            if dst_fldr:
                skimage.io.imsave(dst_fldr + image_fn[:-4] + '_mask.tif', mask.astype('uint8'))
            else:
                skimage.io.imsave(image_fn[:-4] + '_mask.tif', mask.astype('uint8'))
        print('\r', f'Finished {image_fn}')
    print('Done!')
    return


def quarter_divider(image):
    if image.ndim == 3:
        image = np.expand_dims(image, axis=-1)
    dim_length = image.shape[-2]
    if dim_length % 2 == 0:
        half_length = dim_length // 2
        new_im1 = image[:, :half_length, :half_length, :]
        new_im2 = image[:, :half_length, half_length:, :]
        new_im3 = image[:, half_length:, :half_length, :]
        new_im4 = image[:, half_length:, half_length:, :]
        return new_im1, new_im2, new_im3, new_im4
    else:
        print('Image can not be divided evenly!')
        return None


def batch_quarter_divider(batch_fn, dst_fldr=None, save_tif=False):
    for fn in batch_fn:
        f_name = os.path.basename(os.path.normpath(fn))
        img_name, f_ext = os.path.splitext(f_name)
        img_name_path, _ = os.path.splitext(fn)
        if f_ext == '.npy':
            images = quarter_divider(np.load(fn))
        elif f_ext == '.tif':
            images = quarter_divider(skimage.io.imread(fn))
        else:
            print(f'{img_name} is not a npy or tif file! Please provide either a npy or tif file.')
            continue
        for i in range(4):
            if dst_fldr:
                np.save(f'{dst_fldr}{img_name}_quarter{i + 1}.npy', images[i])
                if save_tif:
                    skimage.io.imsave(f'{dst_fldr}{img_name}_quarter{i + 1}.tif', images[i])
            else:
                np.save(f'{img_name_path}_quarter{i + 1}.npy', images[i])
                if save_tif:
                    skimage.io.imsave(f'{img_name_path}_quarter{i + 1}.tif', images[i])


def generate_border(mask, bool_output=False):
    """
    Generates 3 pixel wide border region from binary mask
    :param bool_output:
    :type bool_output:
    :param mask:
    :type mask:
    :return:
    :rtype:
    """
    if bool_output:
        border = skimage.morphology.binary_dilation(mask - skimage.morphology.binary_erosion(mask))
    else:
        border = skimage.morphology.binary_dilation(mask - skimage.morphology.binary_erosion(mask)).astype(mask.dtype)
    return border


def add_border_to_mask(mask):
    border = generate_border(mask, bool_output=True)
    new_mask = np.where(border, 2, mask)
    return new_mask


def np_data_generator(images, labels, batch_size=16, per_channel=False):
    i = 0
    while True:
        if i == (len(images) // batch_size):
            i = 0
            p = np.random.permutation(len(images))
            images = images[p]
            labels = labels[p]
        start = i * batch_size
        stop = start + batch_size
        xbatch = per_sample_tile_normalization(images[start:stop], per_channel=per_channel)
        ybatch = labels[start:stop]
        i += 1
        yield xbatch, ybatch


def np_validation_generator(images, labels, batch_size=16, per_channel=False):
    i = 0
    while True:
        if i == (len(images) // batch_size):
            i = 0
        start = i * batch_size
        stop = start + batch_size
        xbatch = per_sample_tile_normalization(images[start:stop], per_channel=per_channel)
        ybatch = labels[start:stop]
        i += 1
        yield xbatch, ybatch


def np_prediction_generator(images, batch_size=16, per_channel=False):
    if type(images) is str:
        im = np.load(images)
    else:
        im = images
    if len(im) % batch_size == 0:
        for i in range(len(im) // batch_size):
            start = i * batch_size
            stop = start + batch_size
            xbatch = per_sample_tile_normalization(im[start:stop], per_channel=per_channel)
            yield xbatch
    else:
        for i in range((len(im) // batch_size) + 1):
            start = i * batch_size
            if i == (len(im) // batch_size):
                xbatch = per_sample_tile_normalization(im[start:], per_channel=per_channel)
                yield xbatch
            else:
                stop = start + batch_size
                xbatch = per_sample_tile_normalization(im[start:stop], per_channel=per_channel)
                yield xbatch


def aggregate_retinanet_training_dictionaries(dicts):
    aggregated_dictionary = {key: np.concatenate([dic[key] for dic in dicts], axis=0) for key in dicts[0].keys()}
    return aggregated_dictionary


def wsi_generator(WSI, boundingbox, batch_size=1, im_size=512, half_res=True, normalize=True, per_channel=False, two_channel=False):
    if type(WSI) is str:
        wsi = openslide.open_slide(WSI)
    else:
        wsi = WSI
    dim = wsi.dimensions
    counter = 0
    batch = []
    if half_res:
        if two_channel:
            model = half_tile_resolution(im_size, channels=2)
        else:
            model = half_tile_resolution(im_size)
    for i in range(dim[1] // im_size):
        for j in range(dim[0] // im_size):
            # j represent position on x-axis (different from usual which is row #)
            # i represent position on y-axis (different from usual which is column #)
            if two_channel:
                batch.append(np.array(wsi.read_region((j * im_size, i * im_size), 0, (im_size, im_size)))[..., 1:-1])
            else:
                batch.append(np.array(wsi.read_region((j * im_size, i * im_size), 0, (im_size, im_size)))[..., :-1])
            counter += 1
            if counter == batch_size:
                images = np.array(batch)
                counter = 0
                batch = []
                if half_res:
                    images = model.predict(images)
                if normalize:
                    images = per_sample_tile_normalization(np.expand_dims(images, axis=1), per_channel=per_channel)
                else:
                    images = np.expand_dims(images, axis=1)
                batch_dict = {'dat': images}
                for key in boundingbox.params['inputs_shapes'].keys():
                    if 'msk' in key:
                        batch_dict[key] = np.zeros(
                            shape=(batch_size,) + tuple(boundingbox.params['inputs_shapes'][key]))
                yield batch_dict


def local_to_global_coords_retinanet_v2(local_coords, num_of_columns, tile_size):
    # local_coords is a list of arrays which should be in xy format
    global_coords = []
    for i in range(len(local_coords)):
        if len(local_coords[i]) == 0:
            continue
        # get coords of tile
        column = i % num_of_columns
        row = i // num_of_columns
        # pixel range
        c_start = tile_size * column
        r_start = tile_size * row
        for j in range(len(local_coords[i])):
            global_coords.append([c_start + local_coords[i][j, 0], r_start + local_coords[i][j, 1]])
    if len(global_coords) == 0:
        return None
    else:
        return np.array(global_coords)


def cpec_coords_from_anc_v2(anc, num_of_columns, tile_size, half_res=False):
    # anc is a list(2: anchor coords and classes) of a list of arrays (1 array per image tile)
    # anc arrays are local coordinates for the tile
    # convert tile coords to WSI coords
    local_coords = []
    for j in range(len(anc[0])):
        if len(anc[0][j]) != 0:
            y_coords = (anc[0][j][:, 2] + anc[0][j][:, 0]) / 2
            x_coords = (anc[0][j][:, 3] + anc[0][j][:, 1]) / 2
            local_coords.append(np.stack([x_coords, y_coords], axis=1))
        else:
            local_coords.append(anc[0][j])
    wsi_coords = local_to_global_coords_retinanet_v2(local_coords, num_of_columns, tile_size)
    if not half_res:
        return wsi_coords
    elif half_res:
        return (wsi_coords * 2).astype(int)
    else:
        print('half_res parameter must be a boolean!')
        return


def retinanet_prediction_output(WSI, model, boundingbox, batch_size=1, im_size=512, half_res=True, normalize=True,
                                per_channel=False, two_channel=True):
    # want to add path boolean and code load model if model is a filepath
    if type(WSI) is str:
        dim = openslide.open_slide(WSI).dimensions
    else:
        dim = WSI.dimensions
    num_of_images = (dim[0] // im_size)*(dim[1] // im_size)
    if num_of_images % batch_size != 0:
        steps = (num_of_images // batch_size) + 1
    else:
        steps = num_of_images // batch_size
    output = model.predict(wsi_generator(WSI=WSI,
                                         boundingbox=boundingbox,
                                         batch_size=batch_size,
                                         im_size=im_size,
                                         half_res=half_res,
                                         normalize=normalize,
                                         per_channel=per_channel,
                                         two_channel=two_channel),
                           steps=steps)
    output_dic = {name: pred for name, pred in zip(model.output_names, output)}
    return output_dic


def wsi_cpec_generator(WSI, coords, batch_size=16, per_channel=False):
    if type(WSI) is str:
        wsi = openslide.open_slide(WSI)
    else:
        wsi = WSI
    if len(coords) % batch_size == 0:
        for i in range(len(coords) // batch_size):
            start = i * batch_size
            stop = start + batch_size
            xbatch = per_sample_tile_normalization(
                np.expand_dims(wsi_cell_extraction_from_coords_v3(wsi, im_size=64, coords=coords[start:stop], verbose=0), axis=1), per_channel=per_channel)
            yield xbatch
    else:
        for i in range((len(coords) // batch_size) + 1):
            start = i * batch_size
            if i == (len(coords) // batch_size):
                xbatch = per_sample_tile_normalization(
                    np.expand_dims(wsi_cell_extraction_from_coords_v3(wsi, im_size=64, coords=coords[start:], verbose=0), axis=1), per_channel=per_channel)
                yield xbatch
            else:
                stop = start + batch_size
                xbatch = per_sample_tile_normalization(
                    np.expand_dims(wsi_cell_extraction_from_coords_v3(wsi, im_size=64, coords=coords[start:stop], verbose=0), axis=1),
                    per_channel=per_channel)
                yield xbatch


def biondi_prevalence_and_coords(WSI,
                                 retinanet,
                                 classifier,
                                 boundingbox,
                                 batch_size=1,
                                 im_size=512,
                                 half_res=True,
                                 normalize=True,
                                 per_channel=False,
                                 two_channel=True,
                                 iou_nms=0.3):
    if type(WSI) is str:
        wsi = openslide.open_slide(WSI)
    else:
        wsi = WSI
    dim = wsi.dimensions
    if half_res:
        tile_size = im_size // 2
    else:
        tile_size = im_size
    coords = cpec_coords_from_anc_v2(boundingbox.convert_box_to_anc(retinanet_prediction_output(WSI=wsi,
                                                                                                model=retinanet,
                                                                                                boundingbox=boundingbox,
                                                                                                batch_size=batch_size,
                                                                                                im_size=im_size,
                                                                                                half_res=half_res,
                                                                                                normalize=normalize,
                                                                                                per_channel=per_channel,
                                                                                                two_channel=two_channel),
                                                                    iou_nms=iou_nms,
                                                                    apply_deltas=True),
                                     dim[0] // im_size,
                                     tile_size=tile_size,
                                     half_res=half_res)
    prediction_logits = classifier.predict(wsi_cpec_generator(wsi, coords))
    affected_coords = biondi.statistics.sort_affected_coords_from_aipredictions(biondi.statistics.convert_probabilities_to_predictions(prediction_logits), coords)
    prevalence = (len(affected_coords)/len(coords))*100
    return {'coords': coords, 'af_coords': affected_coords, 'prevalence': prevalence}
