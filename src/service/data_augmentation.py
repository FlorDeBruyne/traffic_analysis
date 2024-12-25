import albumentations as A

def get_transformations():
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=640),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=2.5, g_shift_limit=2.5, b_shift_limit=2.5, p=0.25),
        A.RandomBrightnessContrast(p=0.25),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    return train_transform, val_transform