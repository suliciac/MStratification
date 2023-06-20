import numpy as np
import nibabel as nib
from data_utils import ZipSet, ListSet, make_generator, PatchSetPad, normalize


def new_image(args, ims, idx):
    """
    Load input channels for each of the subjects, from csv

    :param idx: int, subject index in csv
    :param args: parser, options load from outside
    :param ims: dataframe, relation of all subject-scans (from csv)
    :return: arr, channels per subj
    """
    ch_im = []
    # load img
    for i in range(args.n_ch):
        im = nib.load(ims.iloc[idx, i])
        im = nib.as_closest_canonical(im)
        im_dat = im.get_fdata().astype('float16')
        # normalize
        norm_im = normalize(im_dat)
        ch_im.append(norm_im)

    return np.array(ch_im)


def new_region(args, ims, idx, mask_dat):
    """
    Load input channels for each of the subjects, from csv

    :param mask_dat: arr, binary ROI mask template
    :param idx: int, subject index in csv
    :param args: parser, options load from outside
    :param ims: dataframe, relation of all subject-scans (from csv)
    :return: arr, channels per subj
    """
    ch_im = []
    # load img
    for i in range(args.n_ch):
        if i == 2:
            i = 4
        im = nib.load(ims.iloc[idx, i])
        im = nib.as_closest_canonical(im)
        im_dat = im.get_fdata().astype('float16')
        #if i != 0:
        norm_im = normalize(im_dat)
        region = norm_im * mask_dat
        ch_im.append(region)
        #else:
        #    norm_im = im_dat
        #    ch_im.append(norm_im)

    return np.array(ch_im)


def load_data(args, data):
    ims, classes = [], []
    if args.roi or args.mask:
        # load roi
        mask = nib.load(args.mask)
        mask = nib.as_closest_canonical(mask)
        mask_dat = mask.get_fdata().astype('float16')
        th_mask = np.zeros_like(mask_dat)
        th_mask[mask_dat >= args.th] = 1

    for ii in range(len(data.iloc[:, 0])):
        if args.roi or args.mask:
            ch = new_region(args, data, ii, th_mask)
        else:
            ch = new_image(args, data, ii)
        ims.append(ch)
        classes.append(data.iloc[ii, 2])

    return ims, classes


def dataloader_mult_strat(args, setdata):
    # load data
    ims, clsses = load_data(args, setdata)

    # centers
    centers = [[args.center] for _ in ims]
    classes = [[clsses[i] for _ in range(0, len(ce))] for i, ce in enumerate(centers)]
    classes = [j for i in classes for j in i]
    # data augmentation
    dict_trans_1 = get_augment_functions()
    if args.nptch1 >= 1:
        new_centers, new_ims, new_classes = [], [], []
        for ii, cls in enumerate(classes):
            if cls == 1:
                if centers[ii] == [(98, 81, 39)]:
                    centers_1 = centers[ii]
                    centers_e1 = add_rand_offset_cond(args, centers_1)
                    cent_e1 = [[tuple(cent)] for cent in centers_e1]
                    new_centers.extend(cent_e1)
                    new_ims.extend([ims[ii] for _ in range(args.nptch1 + 1)])
                    new_classes.extend([classes[ii] for _ in range(args.nptch1 + 1)])

                elif centers[ii] == [(99, 117, 90)]:
                    new_centers.extend([centers[ii] for _ in range(args.nptch1)])
                    new_im = dict_trans_1[0](ims[ii])
                    new_ims.extend([ims[ii], new_im])
                    new_classes.extend([classes[ii] for _ in range(args.nptch1)])
                else:
                    centers_1 = centers[ii]
                    centers_e1 = add_rand_offset_from_centers(args, centers_1)
                    cent_e1 = [[tuple(cent)] for cent in centers_e1]
                    new_centers.extend(cent_e1)
                    new_ims.extend([ims[ii] for _ in range(args.nptch1 + 1)])
                    new_classes.extend([classes[ii] for _ in range(args.nptch1 + 1)])

            else:
                if (centers[ii] == [(99, 117, 90)]) and (np.random.uniform(0., 1.) < args.prob):
                    new_centers.extend([centers[ii] for _ in range(args.nptch1)])
                    new_im = dict_trans_1[0](ims[ii])
                    new_ims.extend([ims[ii], new_im])
                    new_classes.extend([classes[ii] for _ in range(args.nptch1)])

                elif centers[ii] == [(99, 123, 86)]:
                    centers_1 = centers[ii]
                    centers_e1 = add_rand_offset_from_centers(args, centers_1)
                    cent_e1 = [[tuple(centers_e1[cent])] for cent in range(args.nptch2 + 1)]
                    new_centers.extend(cent_e1)
                    new_ims.extend([ims[ii] for _ in range(args.nptch2 + 1)])
                    new_classes.extend([classes[ii] for _ in range(args.nptch2 + 1)])

                elif centers[ii] == [(98, 81, 39)]:
                    centers_1 = centers[ii]
                    centers_e1 = add_rand_offset_cond(args, centers_1)
                    cent_e1 = [[tuple(centers_e1[cent])] for cent in range(args.nptch2 + 1)]
                    new_centers.extend(cent_e1)
                    new_ims.extend([ims[ii] for _ in range(args.nptch2 + 1)])
                    new_classes.extend([classes[ii] for _ in range(args.nptch2 + 1)])

                else:
                    new_centers.append(centers[ii])
                    new_ims.append(ims[ii])
                    new_classes.append(classes[ii])

    else:
        new_centers = centers
        new_ims = ims
        new_classes = classes

    dict_trans = get_augment_functions()
    # patch set
    scan_patch = PatchSetPad(new_ims, args.ps, normalize='none', sampling=None, centers=new_centers, transf_func=dict_trans)
    patch_set = ZipSet([scan_patch, ListSet(new_classes)])

    # make generator
    generator = make_generator(patch_set, batch_size=args.bs, shuffle=True)
    return generator


def dataloader_mult_strat_reg(args, setdata):
    # load data
    ims, clsses = load_data(args, setdata)

    # centers
    centers = [[args.center] for _ in ims]
    classes = [[clsses[i] for _ in range(0, len(ce))] for i, ce in enumerate(centers)]
    classes = [j for i in classes for j in i]
    dict_trans_1 = get_augment_functions()
    if args.nptch1 > 1:
        new_centers, new_ims, new_classes = [], [], []
        for ii, cls in enumerate(classes):
            if cls == 1:
                new_centers.extend([centers[ii] for _ in range(args.nptch1)])
                new_im = dict_trans_1[0](ims[ii])
                new_ims.extend([ims[ii], new_im])
                new_classes.extend([classes[ii] for _ in range(args.nptch1)])
            else:
                if np.random.uniform(0., 1.) < args.prob:
                    new_centers.extend([centers[ii] for _ in range(args.nptch1)])
                    new_im = dict_trans_1[0](ims[ii])
                    new_ims.extend([ims[ii], new_im])
                    new_classes.extend([classes[ii] for _ in range(args.nptch1)])
                else:
                    new_centers.append(centers[ii])
                    new_ims.append(ims[ii])
                    new_classes.append(classes[ii])

    else:
        new_centers = centers
        new_ims = ims
        new_classes = classes
    dict_trans = get_augment_functions()
    # patch set
    scan_patch = PatchSetPad(new_ims, args.ps, normalize='none', sampling=None, centers=new_centers,
                             transf_func=dict_trans)

    patch_set = ZipSet([scan_patch, ListSet(new_classes)])

    # make generator
    if args.action == 'train':
        generator = make_generator(patch_set, batch_size=args.bs, shuffle=True)
        return generator
    else:
        generator = make_generator(patch_set, batch_size=args.bs, shuffle=False)
        return generator


def dataloader_mult_inf(args, setdata):
    ims, clsses = load_data(args, setdata)
    # centers
    centers = [[args.center] for _ in ims]
    classes = [[clsses[i] for _ in range(0, len(ce))] for i, ce in enumerate(centers)]
    classes = [j for i in classes for j in i]

    # patch set
    scan_patch = PatchSetPad(ims, args.ps, normalize='none', sampling=None, centers=centers)
    patch_set = ZipSet([scan_patch, ListSet(classes)])

    # make generator
    generator = make_generator(patch_set, batch_size=1, shuffle=False)
    return generator


def get_augment_functions():
    mean = 0
    var = 0.02
    augment_funcs = {
        0: lambda patch: flipping_axial(patch),
        1: lambda patch, m=mean, v=var: add_noise(patch, mean, var)
    }
    return augment_funcs


def flipping_axial(x):
    x = np.flipud(x)
    return x


def add_noise(x, mean=0, var=0.02):
    # add gausian noise
    if x.shape[0] == 2:
        noise_t1 = np.random.normal(mean, var, x[0].shape)
        noise_fl = np.random.normal(mean, var, x[0].shape)
        noise_fl[x[0] == 0] = 0
        noise_t1[x[1] == 0] = 0
        # renormalize patch (after adding noise)
        x0_norm = normalize(np.array([x[0] + noise_fl]))
        x1_norm = normalize(np.array([x[1] + noise_t1]))
        x = np.concatenate([x0_norm, x1_norm], axis=0)

    elif x.shape[0] == 1:
        noise_ = np.random.normal(mean, var, x[0].shape)
        noise_[x[0] == 0] = 0
        x = np.array([x[0] + noise_])
    else:
        noise_t1 = np.random.normal(mean, var, x[0].shape)
        noise_fl = np.random.normal(mean, var, x[0].shape)
        noise_fl[x[0] == 0] = 0
        noise_t1[x[1] == 0] = 0
        x = np.array([x[0], x[1] + noise_t1, x[2] + noise_fl])

    return x


def add_rand_offset_from_centers(args, centers):
    centers_with_offset = []
    for center in centers:
        centers_with_offset.append(center)
        for _ in range(args.nptch1):
            disp_vox = np.random.randint(2, 12, 3)
            # Positive or negative displacement
            disp_dir = str(np.random.randint(0, 2, 1)[0])
            new_center = [c + disp_vox[i] if disp_dir == '1' else c - disp_vox[i] for i, c in enumerate(center)]
            centers_with_offset.append(new_center)

    return centers_with_offset


def add_rand_offset_cond(args, centers):
    centers_with_offset = []
    for center in centers:
        centers_with_offset.append(center)
        for _ in range(args.nptch1):
            disp_vox = np.random.randint(2, 15, 2)
            disp_vox = np.append(disp_vox, 0)
            # Positive or negative displacement
            disp_dir = str(np.random.randint(0, 2, 1)[0])
            new_center = [c + disp_vox[i] if disp_dir == '1' else c - disp_vox[i] for i, c in enumerate(center)]
            centers_with_offset.append(new_center)

    return centers_with_offset
