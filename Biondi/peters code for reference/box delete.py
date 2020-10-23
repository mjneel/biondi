import os, numpy as np, cv2

class BoundingBox():
    """
    Implementation of bounding-box network parameterizations

    """
    def __init__(self, image_shape, classes=1, c=[3, 4, 5], 
        anchor_shapes=[16, 32, 64], anchor_scales=[0, 1, 2], anchor_ratios=[0.5, 1, 2], 
        iou_upper=0.7, iou_lower=0.3, iou_nms=0.7, box_padding=0, cls_threshold=0.5, separate_maps=True):
        """
        Method to initialize anchors and bounding box parameters

        :params

          (iter)   image_shape     : original 2D image shape
          (int)    classes         : number of non-background classes
          (iter)   c               : feature maps to use; c1 = 1st subsample, c2 = 2nd subsample, etc
          (iter)   anchor_shapes   : base shape of anchors in each feature map
          (iter)   anchor_scales   : scales of each anchor parameterized as 2 ** (i/3)
          (iter)   anchor_ratios   : aspect ratios of each anchor
          (float)  iou_upper       : upper IoU used for pos boxes
          (float)  iou_lower       : lower IoU used for neg boxes
          (float)  iou_nms         : IoU used for non-max supression
          (int)    box_padding     : padding for ground-truth boxes
          (float)  cls_threshold   : threshold to use for classification sigmoid value
          (bool)   separate_maps   : if True, create parameters for each feature map separately
 
        """
        # --- Save params
        self.params = {
            'image_shape': image_shape,
            'classes': classes,
            'c': c,
            'anchor_shapes': anchor_shapes,
            'anchor_scales': anchor_scales,
            'anchor_ratios': anchor_ratios,
            'anchor_levels': None,
            'anchor_gsizes': None,
            'inputs_shapes': None,
            'iou_upper': iou_upper,
            'iou_lower': iou_lower,
            'iou_nms': iou_nms,
            'box_padding': box_padding,
            'cls_threshold': cls_threshold,
            'separate_maps': separate_maps}

        # --- Generate anchors 
        self.init_anchors(**self.params)

    def init_anchors(self, image_shape, c, anchor_shapes, anchor_scales, anchor_ratios, **kwargs):

        # --- Convert Numpy arrays
        image_shape = np.array(image_shape)
        anchor_ratios = np.array(anchor_ratios)
        anchor_scales = np.array(anchor_scales)

        # --- Calculate scales
        anchor_scales = 2 ** (anchor_scales / 3)

        # --- Calculate anchor grid sizes (gsizes) 
        anchor_gsizes = [image_shape / (2 ** n) for n in c]
        anchor_gsizes = [s.astype('int') for s in anchor_gsizes]

        anchors = []
        offsets = []

        for anchor_gsize, anchor_shape in zip(anchor_gsizes, anchor_shapes):
            
            # --- Enumerate heights and widths from scales and ratios
            scales = anchor_shape * anchor_scales
            hs = scales.reshape(-1, 1) / np.sqrt(anchor_ratios).reshape(1, -1)
            ws = scales.reshape(-1, 1) * np.sqrt(anchor_ratios).reshape(1, -1)

            hs = hs.ravel()
            ws = ws.ravel()

            # --- Enumerate shifts in feature space
            stride = image_shape / anchor_gsize
            shifts_h = np.arange(0, anchor_gsize[0]) * stride[0]
            shifts_w = np.arange(0, anchor_gsize[1]) * stride[1]
            shifts_y, shifts_x = np.meshgrid(shifts_h, shifts_w, indexing='ij')

            # --- Enumerate combinations of shifts, widths, and heights
            box_hs, box_centers_y = np.meshgrid(hs, shifts_y)
            box_ws, box_centers_x = np.meshgrid(ws, shifts_x)

            # Reshape to get a list of (y, x) and a list of (h, w)
            box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
            box_sizes = np.stack([box_hs, box_ws], axis=2).reshape([-1, 2])
            offsets.append(np.concatenate([box_centers, box_sizes], axis=1))

            # Convert to corner coordinates (y1, x1, y2, x2)
            boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)
            boxes = self.clip_boxes(boxes)
            anchors.append(boxes)

        levels = [anchor.shape[0] for anchor in anchors] 
        levels = [sum(levels[:n + 1]) for n in range(len(levels))]
        self.params['anchor_levels'] = levels
        self.params['anchor_gsizes'] = [list(g) for g in anchor_gsizes] 

        self.anchors = np.concatenate(anchors)
        self.offsets = np.concatenate(offsets)

        # --- Record anchor areas
        self.anchor_areas = self.calculate_anchor_areas(self.anchors)

        # --- Record cls / reg shapes
        self.calculate_inputs_shapes()

    def clip_boxes(self, boxes):

        image_shape = self.params['image_shape']
        boxes = boxes.clip(min=0)
        boxes[:, 0] = boxes[:, 0].clip(max=image_shape[0])
        boxes[:, 1] = boxes[:, 1].clip(max=image_shape[1])
        boxes[:, 2] = boxes[:, 2].clip(max=image_shape[0])
        boxes[:, 3] = boxes[:, 3].clip(max=image_shape[1])

        return boxes

    def calculate_inputs_shapes(self):
        """
        Method to calculate cls / reg shapes and keys

        """
        A = len(self.params['anchor_scales']) * len(self.params['anchor_ratios'])
        K = self.params['classes']
        p = {}

        if self.params['separate_maps']:
            for c, s in zip(self.params['c'], self.params['anchor_gsizes']):

                p['cls-c{}'.format(c)] = [1] + s + [int(A * K)]
                p['reg-c{}'.format(c)] = [1] + s + [int(A * 4)]
                p['cls-c{}-msk'.format(c)] = [1] + s + [int(A * K)]
                p['reg-c{}-msk'.format(c)] = [1] + s + [int(A * 4)]

        else:
            p['cls'] = [1] + [self.anchors.shape[0]] + [K, 1]
            p['reg'] = [1] + [self.anchors.shape[0]] + [4, 1]
            p['cls-msk'] = [1] + [self.anchors.shape[0]] + [K, 1]
            p['reg-msk'] = [1] + [self.anchors.shape[0]] + [4, 1]

        self.params['inputs_shapes'] = p

    # ============================================================================
    # CONVERSION TOOLS 
    # ============================================================================
    # 
    # Generic methods to convert various box parameterizations:
    #
    #   * box : cls, reg, cls-msk and reg-msk components
    #   * msk : full H x W image mask (each blob considered a box source)
    #   * anc : anchors and offsets
    # 
    # ============================================================================

    def convert_msk_to_box(self, msk):
        """
        Method to convert msk to box parameterization

        NOTE, msk may be either:

          * 4D tensor ==> returns single-slice box parameterization
          * 5D tensor ==> returns multi-slice box parameterizations

        :params

          (np.ndarray) msk

        """
        # TODO: consider vectorization this code

        if msk.ndim < 5:
            return self.convert_msk_to_box_single(msk)

        axis = 0 if msk.shape[0] > 1 else 1
        assert msk.shape[axis] > 1

        # --- Run N-number of conversions
        box = {k: [] for k in self.params['inputs_shapes']}

        for i in range(msk.shape[axis]):
            b = self.convert_msk_to_box_single(msk=msk[i] if axis == 0 else msk[:, i])
            for k in box:
                box[k].append(b[k])

        return {k: np.stack(v, axis=axis) for k, v in box.items()}

    def convert_anc_to_box(self, anchors, classes):
        """
        Method to convert anc to box parameterization

        NOTE, anchors may be either:

          * 4D tensor ==> returns single-slice box parameterization
          * 5D tensor ==> returns multi-slice box parameterizations

        :params

          (np.ndarray) anchors (4D or 5D) 
          (np.ndarray) classes (1D or 2D) 

        """
        # TODO: consider vectorization this code

        anchors = np.array(anchors)
        classes = np.array(classes)

        if anchors.ndim < 5:
            return self.convert_anc_to_box_single(anchors, classes)

        axis = 0 if anchors.shape[0] > 1 else 1
        assert anchors.shape[axis] > 1

        # --- Run N-number of conversions
        box = {k: [] for k in self.params['inputs_shapes']}

        for i in range(anchors.shape[axis]):
            a = anchors[i] if axis == 0 else anchors[:, i]
            b = self.convert_anc_to_box_single(anchors=a, classes=classes[i])
            for k in box:
                box[k].append(b[k])

        return {k: np.stack(v, axis=axis) for k, v in box.items()}

    def convert_msk_to_box_single(self, msk):
        """
        Method to convert msk to box parameterization

        """
        # --- Convert msk to anchors / classes
        anchors, classes = self.calculate_boxes(msk)

        return self.convert_anc_to_box_single(anchors, classes)

    def convert_anc_to_box_single(self, anchors, classes):
        """
        Method to convert anc to box parameterization

        """
        cls, msk, dst = self.calculate_gt_cls(anchors, classes)
        reg = self.calculate_gt_reg(dst)

        # --- Determine parameterization
        if self.params['separate_maps']:
            return self.vec_to_map(cls=cls, reg=reg, msk=msk)

        else:
            return {**{'cls': cls, 'reg': reg}, **msk}

    def calculate_gt_cls(self, anchors, classes):
        """
        Method to calculate which ground-truth anchors are positive 

        """
        cls = np.zeros(self.anchors.shape[0], dtype='uint8')
        dst = np.zeros(self.anchors.shape, dtype='float32')

        cls_msk = np.zeros(self.anchors.shape[0], dtype='uint8')
        reg_msk = np.zeros(self.anchors.shape, dtype='float32')

        for a, c in zip(anchors, classes): 

            ious = self.calculate_ious(box=a)

            # --- Adjust IOUs 
            iou_upper = min(self.params['iou_upper'], max(ious))
            iou_lower = min(self.params['iou_lower'], iou_upper) 

            # --- Determine anchors with required IOU 
            pos = (ious >= iou_upper)
            ind = (ious >  iou_lower) & (ious < iou_upper)

            cls[pos] = c 
            dst[pos] = a

            cls_msk[ind] = c 

        # --- Create msk for reg
        reg_msk[cls > 0] = 1 

        # --- One-hot vector cls
        if self.params['classes'] > 1:
            cls = np.eye(self.params['classes'] + 1)[cls]
            cls = cls[:, 1:]

            cls_msk = np.eye(self.params['classes'] + 1)[cls_msk]
            cls_msk = cls_msk[:, 1:]

        # --- Invert cls_msk
        cls_msk = (1 - cls_msk).astype('float32')

        msk = {'cls-msk': cls_msk, 'reg-msk': reg_msk}

        return cls, msk, dst 

    def calculate_gt_reg(self, anchors):
        """
        Method to calculate ground-truth deltas 

        """
        return self.calculate_deltas(anchors=anchors)

    def convert_box_to_anc(self, box, apply_deltas=True, iou_nms=None, cls_threshold=None):
        """
        Method to convert box parameterizations to anchors 

        NOTE, box values may be either:

          * 4D tensor ==> assumed to be single box parameterization
          * 5D tensor ==> assumed to be multiple boxes

        """
        # TODO: consider vectorization this code

        shape = next(iter(box.values())).shape

        kwargs = {
            'apply_deltas': apply_deltas,
            'iou_nms': iou_nms,
            'cls_threshold': cls_threshold}

        if len(shape) < 5:
            return self.convert_box_to_anc_single(box, **kwargs)

        axis = 0 if shape[0] > 1 else 1
        assert shape[axis] > 1

        # --- Run N-number of conversions
        anchors = []
        classes = []

        for i in range(shape[axis]):
            b = {k: v[i] if axis == 0 else v[:, i] for k, v in box.items()}
            a, c = self.convert_box_to_anc_single(b, **kwargs)
            anchors.append(a)
            classes.append(c)

        return anchors, classes 

    def convert_box_to_msk(self, box, apply_deltas=True, iou_nms=None, cls_threshold=None, random_values=True):
        """
        Method to convert box parameterizations to msk

        NOTE, box values may be either:

          * 4D tensor ==> assumed to be single box parameterization
          * 5D tensor ==> assumed to be multiple boxes

        """
        # TODO: consider vectorization this code

        shape = next(iter(box.values())).shape

        kwargs = {
            'apply_deltas': apply_deltas,
            'iou_nms': iou_nms,
            'cls_threshold': cls_threshold,
            'random_values': random_values}

        if len(shape) < 5:
            return self.convert_box_to_msk_single(box, **kwargs)

        axis = 0 if shape[0] > 1 else 1
        assert shape[axis] > 1

        # --- Run N-number of conversions
        msk = []

        for i in range(shape[axis]):
            b = {k: v[i] if axis == 0 else v[:, i] for k, v in box.items()}
            msk.append(self.convert_box_to_msk_single(b, **kwargs))

        return np.stack(msk, axis=axis)

    def convert_box_to_anc_single(self, box, apply_deltas=True, iou_nms=None, cls_threshold=None):
        """
        Method to convert box parameterizations to anchors

        """
        # --- Convert parameterization if needed
        if self.params['separate_maps']:
            box = self.map_to_vec(box)

        if apply_deltas:
            anchors, _ = self.apply_deltas(box['reg'])
        else:
            anchors = self.anchors

        # --- Calculate classification results
        scores = 1 / (1 + np.exp(box['cls'] * -1))

        cls_threshold = cls_threshold or self.params['cls_threshold']
        classes = scores > cls_threshold
        indices = np.nonzero(np.max(classes, axis=1))[0]

        anchors = anchors[indices]
        classes = classes[indices]

        # --- Apply NMS
        if apply_deltas:
            scores = np.max(scores[indices], axis=1)
            fg = self.apply_nms(anchors, scores, iou_nms)
            anchors = anchors[fg]
            classes = classes[fg]

        return anchors, classes

    def convert_box_to_msk_single(self, box, apply_deltas=True, iou_nms=None, cls_threshold=None, random_values=True):
        """
        Method to convert box parameterizations to msk

        """
        msk = np.zeros(self.params['image_shape'], dtype='uint8')

        anchors, classes = self.convert_box_to_anc_single(box, 
                apply_deltas=apply_deltas,
                iou_nms=iou_nms,
                cls_threshold=cls_threshold)

        for i in range(anchors.shape[0]):
            val = None if random_values else np.argmax(classes[i]) + 1
            msk = self.draw_box(msk, anchor=anchors[i], value=val)

        return np.expand_dims(np.expand_dims(msk, -1), 0) 

    def vec_to_map(self, cls, reg, msk):
        """
        Method to convert vectorized anchors / deltas to separate FPN maps

        """
        d = {}
        s = self.params['inputs_shapes']

        for i, c in enumerate(self.params['c']):

            lo = self.params['anchor_levels'][i - 1] if i > 0 else 0
            hi = self.params['anchor_levels'][i]
            c = str(c)

            d['cls-c{}'.format(c)] = cls[lo:hi].reshape(s['cls-c{}'.format(c)])
            d['reg-c{}'.format(c)] = reg[lo:hi].reshape(s['reg-c{}'.format(c)])
            d['cls-c{}-msk'.format(c)] = msk['cls-msk'][lo:hi].reshape(s['cls-c{}-msk'.format(c)])
            d['reg-c{}-msk'.format(c)] = msk['reg-msk'][lo:hi].reshape(s['reg-c{}-msk'.format(c)])

        return d

    def map_to_vec(self, box):
        """
        Method to convert separate FPN maps to vectorized anchors / deltas

        """
        d = {}

        for k, c in zip(['cls', 'reg'], [self.params['classes'], 4]):

            keys = sorted([k_ for k_ in box if k_[:3] == k and k_[-3:] != 'msk'])
            d[k] = np.concatenate([box[k_].reshape(-1, c) for k_ in keys], axis=0)

            if k == 'cls' and c > 1:
                bg = np.sum(d[k], axis=1) == 0
                d[k] = np.argmax(d[k], axis=1) + 1
                d[k][bg] = 0

        return d

    # ============================================================================
    # VECTORIZED BOX MATH
    # ============================================================================

    def calculate_boxes(self, msk):
        """
        Method to generate bounding box of msk

        """
        anchors = []
        classes = []

        msk = msk.squeeze()
        assert msk.ndim == 2

        count, labels = cv2.connectedComponents(
            image=(msk > 0).astype('uint8'),
            connectivity=4,
            ltype=cv2.CV_32S)

        for i in range(1, count):

            l = labels == i
            anchors.append(self.calculate_box_single(l))
            classes.append(msk[l][0])

        return np.array(anchors), np.array(classes)

    def calculate_box_single(self, msk):

        points = []

        for axis in [1, 0]:

            reduced = np.nonzero(np.sum(msk, axis=axis))[0]
            e0, e1 = reduced[0], reduced[-1]
            epoints = np.array([
                e0 - self.params['box_padding'], 
                e1 + self.params['box_padding'] + 1])
            epoints = epoints.clip(min=0, max=msk.shape[int(not axis)])
            points.append(epoints)

        ys, xs = points
        y1, y2 = ys
        x1, x2 = xs

        return np.array([y1, x1, y2, x2])

    def calculate_anchor_areas(self, anchors):

        y = anchors[:, 2] - anchors[:, 0]
        x = anchors[:, 3] - anchors[:, 1]

        return y * x 

    def calculate_ious(self, box, anchors=None, anchor_areas=None):
        """
        Method to generate all anchor IOUs for a given bounding box

        :params

          (np.array) box 
          (np.array) anchors

        :return

          (np.array) ious : matrix with anchors.shape[0] entries

        """
        if anchors is None:
            anchors = self.anchors
            anchor_areas = self.anchor_areas
        else:
            if anchor_areas is None:
                anchor_areas = self.calculate_anchor_areas(anchors)

        y1 = np.maximum(box[0], anchors[:, 0])
        x1 = np.maximum(box[1], anchors[:, 1])
        y2 = np.minimum(box[2], anchors[:, 2])
        x2 = np.minimum(box[3], anchors[:, 3])

        yd = y2 - y1
        xd = x2 - x1
        inter = abs(yd * xd)
        inter[(yd < 0) | (xd < 0)] *= -1
        union = anchor_areas + (
            ((box[2] - box[0]) * (box[3] - box[1])) - inter)

        return (inter / union).clip(min=0)

    def calculate_overlap(self, msk, box):
        """
        Method to generate overlap (different than IOU) for mask and given box

        """
        num = np.count_nonzero(msk[
            box[0]:box[2],
            box[1]:box[3]])

        den = (box[2] - box[0]) * (box[3] - box[1])

        return num / den

    def calculate_deltas(self, anchors, offsets=None, epsilon=1e-6):
        """
        Method to generate deltas that transform (src) offsets to (dst) anchors

        Note: any anchors where entire == 0 are set to delta == 0

        :return

          (np.array) deltas

        """
        if offsets is None:
            offsets = self.offsets

        inds = np.sum(np.abs(anchors), axis=1) == 0

        # --- Source coordinates
        y, x, h, w = np.split(offsets, [1, 2, 3], axis=-1)

        # --- Target coordinates
        y1_, x1_, y2_, x2_ = np.split(anchors, [1, 2, 3], axis=-1)

        y_ = (y1_ + y2_) / 2
        x_ = (x1_ + x2_) / 2
        h_ = y2_ - y1_
        w_ = x2_ - x1_

        h_[inds] = epsilon 
        w_[inds] = epsilon 

        # --- Parameterized delta
        deltas = [
            (y_ - y) / h, 
            (x_ - x) / w,
            np.log(h_ / (h + epsilon)), 
            np.log(w_ / (w + epsilon))]
        deltas = np.concatenate(deltas, axis=-1)

        # --- Set empty anchors to zero
        deltas[inds] = 0

        return deltas.astype('float32')

    def apply_deltas(self, deltas, offsets=None):
        """
        Method to apply deltas to anchors and offsets

        :returns

          (np.array) anchors
          (np.array) offsets

        """
        if offsets is None:
            offsets = self.offsets.copy()

        offsets[..., :2] = (deltas[..., :2] * offsets[..., 2:] + offsets[..., :2])
        offsets[..., 2:] = (np.exp(deltas[..., 2:]) * (offsets[..., 2:] + 1e-6))

        anchors = [
            offsets[..., 0] - offsets[..., 2] * 0.5,
            offsets[..., 1] - offsets[..., 3] * 0.5,
            offsets[..., 0] + offsets[..., 2] * 0.5,
            offsets[..., 1] + offsets[..., 3] * 0.5]

        anchors = np.stack(anchors, axis=-1)

        return anchors.astype('float32'), offsets.astype('float32')

    def apply_nms(self, anchors, scores, iou_nms=None):
        """
        Method to apply greedy non-maxmimum suppression to anchors

          (1) Find anchor with highest score
          (2) Remove all anchors with IOU > iou_nms
          (3) Find remaining anchor with next highest score and iterate ... 

        :params

          (float) iou_nms : if provided, IOU threshold (else use self.params['iou_nms'])

        :return

          (np.array) fg : indices for foreground (not suppressed)

        """
        iou_nms = iou_nms or self.params['iou_nms']
        anchor_areas = self.calculate_anchor_areas(anchors)

        fg = []

        while scores.any():

            # --- Choose anchor with the remaining highest score
            argmax = np.argmax(scores)
            fg.append(argmax)
            scores[argmax] = 0

            if not scores.any():
                break

            # --- Calculate IOUs on selected box and remaining anchors
            ious = self.calculate_ious(box=anchors[fg[-1]], anchors=anchors, anchor_areas=anchor_areas)

            # --- Remove all IOUs > threshold
            scores[ious > iou_nms] = 0

        return np.sort(fg).astype('int')

    def draw_box(self, msk, anchor, value=None, fill=False, stroke=1):
        """
        Method to create outline/filled box on msk

        :params

          (int) value : value to draw; if None, random value from [1, 10]

        """
        value = value or np.random.randint(1, 10)

        anchor = anchor.astype('int').clip(min=0, max=msk.shape[0] - stroke)
        y1, x1, y2, x2 = anchor 

        if fill:
            msk[y1:y2, x1:x2] = value

        else:
            msk[y1:y1 + stroke, x1:x2] = value
            msk[y2:y2 + stroke, x1:x2] = value
            msk[y1:y2, x1:x1 + stroke] = value
            msk[y1:y2, x2:x2 + stroke] = value

        return msk 

    # ============================================================================
    # GENERATORS and INPUTS
    # ============================================================================

    def generator(self, G, msk, **kwargs):

        for xs, ys in G:

            y = ys.pop(msk)
            box = self.convert_msk_to_box(msk=y)

            xs.update({k: box[k] for k in box if k[-3:] == 'msk'})
            ys.update({k: box[k] for k in box if k[-3:] != 'msk' and k[:3] in ['cls', 'reg']})

            yield xs, ys

    def create_generators(self, train, valid, msk='lbl', **kwargs):

        gt = self.generator(train, msk, **kwargs)
        gv = self.generator(valid, msk, **kwargs)

        return gt, gv

    def get_inputs(self, inputs, Input):
        """
        Method to create dictionary of Keras-type Inputs(...) based on self.specs

        """
        shapes = {k: v for k, v in self.params['inputs_shapes'].items() if k[-3:] == 'msk'}

        msks = {k: Input(
                shape=[None] + v[1:], 
                dtype='float32', 
                name=k) for k, v in shapes.items()}

        inputs.update(msks)

        return inputs

# =============================================================
# from jarvis.tools import show
# bb = BoundingBox(image_shape=(256, 256), 
#     iou_upper=0.5)
# =============================================================
# msk = np.zeros((256, 256))
# msk[100:140, 100:140] = 1
# p = bb.convert_msk_to_box(msk)
# m = bb.convert_box_to_msk(p, apply_deltas=False)
# =============================================================
# anchors = [[100, 100, 140, 140]]
# classes = [1]
# p_ = bb.convert_anc_to_box(anchors, classes)
# a, c = bb.convert_box_to_anc(p_, apply_deltas=True)
# =============================================================
