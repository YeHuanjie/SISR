#%%
# coding=utf-8
import os
import IQA
import cv2

upscale_factor = 4

# save_root need to change for different method
save_root = 'srgan'
save_hr = os.path.join(save_root, 'origin')
save_lr = os.path.join(save_root, save_root)

# assess
f_path = os.path.join(save_root, 'score.txt')
f = open(f_path, "a+")
print('file', 'y_psnr', 'all_psnr', 'ssim', 'msssim', 'gmsd', 'mdsi', 'gcsd', file=f)
f.close()

for file in os.listdir(save_hr):
    origin_img = cv2.imread(os.path.join(save_hr, file))
    sr_img = cv2.imread(os.path.join(save_lr, save_root + '_' + file))

    h0, w0, _ = origin_img.shape
    h0 = h0 // upscale_factor * upscale_factor
    w0 = w0 // upscale_factor * upscale_factor
    origin_img = origin_img[0: h0, 0: w0, :]

    # sr score
    sr_y_psnr = round(IQA.psnr_y(origin_img, sr_img), 4)
    sr_psnr = round(IQA.psnr(origin_img, sr_img), 4)
    sr_ssim = round(IQA.ssim(origin_img, sr_img), 4)
    sr_msssim = round(IQA.ms_ssim(origin_img, sr_img), 4)
    sr_gmsd = round(IQA.gmsd(origin_img, sr_img), 4)
    sr_mdsi = round(IQA.mdsi(origin_img, sr_img), 4)
    sr_gcsd = round(IQA.gcsd(origin_img, sr_img), 4)

    f = open(f_path, "a+")
    print(save_root + '_' + file, sr_y_psnr, sr_psnr, sr_ssim, sr_msssim, sr_gmsd, sr_mdsi, sr_gcsd, file=f)
    f.close()
