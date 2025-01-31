import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.message import EmailMessage


def window(img, window_center=-500, window_width=1400):
        
    low = img.min()
    high = img.max()

    low = window_center - window_width // 2
    high = window_center + window_width // 2

    img[img < low] = low
    img[img > high] = high
    img = ((img - low) / (high - low) * 255).astype('uint8')
    return img
def overlay(image, label):
    img = image[..., None]
    img3 = np.concatenate([img, img, img], 2)
    #80ae80 is the color of rul
    img3[label == 1, ...] = img3[label == 1, ...] * 0.6 + np.array([128, 174, 128]) * 0.4
    #f1d691 is the color of rml
    img3[label == 2, ...] = img3[label == 2, ...] * 0.6 + np.array([145, 214, 241]) * 0.4
    #b17a65 is the color of rll
    img3[label == 3, ...] = img3[label == 3, ...] * 0.6 + np.array([101, 122, 177]) * 0.4
    #6fb8d2 is the color of lul
    img3[label == 4, ...] = img3[label == 4, ...] * 0.6 + np.array([210, 184, 111]) * 0.4
    #d8654f is the color of lll
    img3[label == 5, ...] = img3[label == 5, ...] * 0.6 + np.array([79, 101, 216]) * 0.4
    # #dd8265 is the color of bronchi
    img3[label == 6, ...] = img3[label == 6, ...] * 0.6 + np.array([221, 130, 101]) * 0.4
    return img3.astype('uint8')

def convert_lola_to_arr(arr):
    arr2 = np.zeros_like(arr)
    arr2[arr == 10] = 1
    arr2[arr == 11] = 2
    arr2[arr == 20] = 3
    arr2[arr == 21] = 4
    arr2[arr == 22] = 5
    return arr2

def get_4d_image_from_16bit(image, label, label2=None):
    img_16 = (image[..., None] + 4096).astype('uint16')
    img_l8 = img_16 % 256
    img_h8 = img_16 / 256
    if label2 is None:
        label2 = np.zeros_like(label)
    img_4 = np.concatenate([img_l8, img_h8, label[..., None], label2[..., None]], 2)
    return img_4.astype('uint8')

def get_3d_image_from_16bit(image, label=None):
    img_16 = (image[..., None] + 4096).astype('uint16')
    img_l8 = img_16 % 256
    img_h8 = img_16 / 256
    if label is None:
        label = np.zeros_like(image)
    img_3 = np.concatenate([img_l8, img_h8, label[..., None]], 2)
    return img_3.astype('uint8')

def make_pngs_of_lobes(patient_id, unique_id, niftis_folder, outputs_folder, pngs_folder, models=['lungs'], lola=False):

    img = sitk.ReadImage(f"{niftis_folder}/images/{patient_id}_{unique_id}_0000.nii.gz")
    img_array = sitk.GetArrayFromImage(img)

    # label_array = np.zeros(img_array.shape).astype('uint16')
    # k = 1
    # for model in ['lobes', 'lungs', 'infiltration']:
    #     if os.path.exists(f"{outputs_folder}/{model}/{patient_id}_{unique_id}.nii.gz"):
    #         label = sitk.ReadImage(f"{outputs_folder}/{model}/{patient_id}_{unique_id}.nii.gz")
    #         label_array = sitk.GetArrayFromImage(label) * k + label_array
    #     k = k * 10

    # label_array2 = None
    # model = 'ild'
    # if os.path.exists(f"{outputs_folder}/{model}/{patient_id}_{unique_id}.nii.gz"):
    #     label2 = sitk.ReadImage(f"{outputs_folder}/{model}/{patient_id}_{unique_id}.nii.gz")
    #     label_array2 = sitk.GetArrayFromImage(label2)

    # img_array = np.rot90(img_array, axes=(1,0), k=2)
    # label_array = np.rot90(label_array, axes=(1,0), k=2)
    # if label_array2 is not None:
    #     label_array2 = np.rot90(label_array2, axes=(1,0), k=2)   


    # for i in range(img_array.shape[0]):
    #     os.system(f"mkdir -p {pngs_folder}/{patient_id}_{unique_id}/")
    #     if label_array2 is None:
    #         img = get_4d_image_from_16bit(img_array[i, ...], label_array[i, ...])
    #     else:
    #         img = get_4d_image_from_16bit(img_array[i, ...], label_array[i, ...], label_array2[i, ...])
    #     cv2.imwrite(f"{pngs_folder}/{patient_id}_{unique_id}/{i}.png", img)

    img_array = np.rot90(img_array, axes=(1,0), k=2)
    os.system(f"mkdir -p {pngs_folder}/{patient_id}_{unique_id}/")
    for i in range(img_array.shape[0]):
        img = get_3d_image_from_16bit(img_array[i, ...])
        cv2.imwrite(f"{pngs_folder}/{patient_id}_{unique_id}/{i}.png", img)

    for model in models:
        if os.path.exists(f"{outputs_folder}/{model}/{patient_id}_{unique_id}.nii.gz"):
            label = sitk.ReadImage(f"{outputs_folder}/{model}/{patient_id}_{unique_id}.nii.gz")
            label_array = sitk.GetArrayFromImage(label)
            label_arr = np.rot90(label_array, axes=(1,0), k=2)
            d, h, w = label_arr.shape
            img_2d = np.zeros((d * h, w), dtype=np.uint8)
            for i in range(d):
                img_2d[i * h:(i + 1) * h, :] = label_arr[i, ...]
            cv2.imwrite(f"{pngs_folder}/{patient_id}_{unique_id}/output_{model}.png", img_2d)



def send_prediction_mail(patient_id, unique_id, access_key, mail_to, config):

    smtp_user = config['smtp_user']
    smtp_password = config['smtp_password']
    smtp_server = config['smtp_server']
    smtp_port = config['smtp_port']
    client_ip = config['client_ip']
    client_port = config['client_port']
    server_ip = config['server_ip']
    server_port = config['server_port']

    msg = EmailMessage()
    content = """
    Patient ID: {}
    Unique ID: {}
    Status: Prediction Done
    View Link: "http://{}:{}/app/prediction/{}/{}/{}"
    """.format(patient_id, unique_id, client_ip, client_port , patient_id, unique_id, access_key)
    msg.set_content(content)
    msg['Subject'] = 'Prediction for {}_{}'.format(patient_id, unique_id)
    msg['From'] = smtp_user
    msg['To'] = mail_to


    server = smtplib.SMTP(smtp_server, smtp_port)
    server.connect(smtp_server, smtp_port)
    server.ehlo()
    server.starttls()
    server.login(smtp_user, smtp_password)
    server.send_message(msg)
    server.close()

def send_metrics_mail(unique_id, metrics, model, model_labels, mail_to, config):

    smtp_user = config['smtp_user']
    smtp_password = config['smtp_password']
    smtp_server = config['smtp_server']
    smtp_port = config['smtp_port']
    client_ip = config['client_ip']
    client_port = config['client_port']
    server_ip = config['server_ip']
    server_port = config['server_port']

    metric_names = [
        'Recall',
        'Precision',
        'Dice',
        'Jaccard',
    ]

    # convert list of dicts to single dict
    metrics = {k: v for d in metrics for k, v in d.items()}


    ## create metrics table using html
    html_table = """
    <table border="1">
    <tr>
    <th>Metric</th>
    """
    for metric_name in metric_names:
        html_table += f"<th>{metric_name}</th>"
    
    html_table += """
    </tr>
    """

    for label in model_labels:
        html_table += """
        <tr>
        <td>{}</td>
        """.format(label)
        for i, metric_name in enumerate(metric_names):
            html_table += f"<td>{metrics[label][i]}</td>"
        html_table += """
        </tr>
        """

    html_table += """
    </table>
    """

    html = """
    <html>
    <body>
    <h1>Metrics for {}</h1>
    <h2>Model: {}</h2>
    {}
    </body>
    </html>
    """.format(unique_id, model, html_table)

    msg = MIMEMultipart()
    msg['Subject'] = 'Metrics for {}'.format(unique_id)
    msg['From'] = smtp_user
    msg['To'] = mail_to

    msg.attach(MIMEText(html, 'html'))


    server = smtplib.SMTP(smtp_server, smtp_port)
    server.connect(smtp_server, smtp_port)
    server.ehlo()
    server.starttls()
    server.login(smtp_user, smtp_password)
    server.send_message(msg)
    server.close()

def format_date(d):
    yy = d[:4]
    mm = d[4:6]
    dd = d[6:]
    return f'{dd}-{mm}-{yy}'
    
if __name__ == '__main__':
    make_pngs_of_lobes('lola11-46', True)