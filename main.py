from lungmask import mask
import SimpleITK as sitk
import os
import datetime

def work(src_path): 
    input_image = sitk.ReadImage(src_path)
    img = sitk.GetArrayFromImage(input_image)
    spacing = input_image.GetSpacing()
    direction = input_image.GetDirection()
    origin = input_image.GetOrigin()

    model = mask.get_model('unet', 'LTRCLobes', 'r231.pth')
    segmentation = mask.apply(input_image, model, batch_size = 20)
    segmentation[segmentation == 2] = 1

    segmented_name = os.path.basename(src_path).replace('.nii.gz', '_segmented.nii.gz')
    contour_name = os.path.basename(src_path).replace('.nii.gz', '_contour.nii.gz')
    
    out_label = sitk.GetImageFromArray(segmentation * img)
    out_label.SetSpacing(spacing)
    out_label.SetOrigin(origin)
    out_label.SetDirection(direction)
    sitk.WriteImage(out_label, segmented_name)
    
    segmented_img = sitk.GetImageFromArray(segmentation)
    sitk.WriteImage(segmented_img, contour_name)

if __name__ == "__main__":
     __directory__ = "input"
     for root, dirs, files in os.walk(__directory__):
         for file in files:
                if file.endswith('.nii.gz'):
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)
                    if size > 5 * 1024 * 1024: 
                        work(file_path)
                    print(time.strftime('%H:%M:%S') + " ok: " + file_path)
