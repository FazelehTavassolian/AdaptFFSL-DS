from torchvision import transforms


def mk_train_transform(args) -> list:
    trans = [transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(args.crop_x, args.crop_y))]

    if args.color:
        trans.append(transforms.ColorJitter(brightness=args.color_br, contrast=args.color_co, saturation=args.color_st))

    if args.h_flip:
        trans.append(transforms.RandomHorizontalFlip(p=args.h_flip_p))

    if args.v_flip:
        trans.append(transforms.RandomVerticalFlip(p=args.v_flip_p))

    if args.rotate:
        trans.append(transforms.RandomRotation(args.rotate_angle))

    if args.shear:
        trans.append(transforms.RandomAffine(args.rotate_angle))

    return trans
