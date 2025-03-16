import argparse
import os
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import cv2
from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet batch evaluation', add_help=False)
    
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save results')
    parser.add_argument('--weight_path', default='./ckpt/latest.pth',
                        help='path where the trained weights saved')
    parser.add_argument('--input_dir', default='./teste_real',
                        help='directory containing input images')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='the gpu used for evaluation')
    parser.add_argument('--confidence_threshold', default=0.3, type=float,
                        help='threshold for filtering detections')
    parser.add_argument('--multi_scale', action='store_true',
                        help='enable multi-scale detection')
    parser.add_argument('--scale_factors', default=[0.8, 1.0, 1.2], type=float, nargs='+',
                        help='scale factors for multi-scale detection')
    
    return parser

def process_image_multi_scale(model, transform, img_raw, scale_factor, device):
    # Redimensionar a imagem de acordo com o fator de escala
    width, height = img_raw.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Arredondar para múltiplos de 128
    new_width = (new_width // 128) * 128
    new_height = (new_height // 128) * 128
    
    img_scaled = img_raw.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img = transform(img_scaled)
    
    # Preparar para inferência
    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    
    # Executar inferência
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    
    # Ajustar as coordenadas de volta para a escala original
    outputs_points = outputs_points / scale_factor
    
    return outputs_points, outputs_scores

def process_image(model, transform, img_path, device, output_dir, confidence_threshold=0.3, multi_scale=False, scale_factors=[0.8, 1.0, 1.2]):
    # Carregar a imagem
    img_raw = Image.open(img_path).convert('RGB')
    width, height = img_raw.size
    
    all_points = []
    all_scores = []
    
    if multi_scale:
        # Processar em múltiplas escalas
        for scale_factor in scale_factors:
            points, scores = process_image_multi_scale(model, transform, img_raw, scale_factor, device)
            all_points.append(points)
            all_scores.append(scores)
        
        # Concatenar resultados de todas as escalas
        outputs_points = torch.cat(all_points)
        outputs_scores = torch.cat(all_scores)
    else:
        # Processar em escala única
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img_raw = img_raw.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img = transform(img_raw)
        
        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)
        
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
    
    # Filtrar predições usando NMS (Non-Maximum Suppression)
    mask = outputs_scores > confidence_threshold
    points = outputs_points[mask].detach().cpu().numpy()
    scores = outputs_scores[mask].detach().cpu().numpy()
    
    # Aplicar NMS
    keep = nms(points, scores, threshold=0.3)
    points = points[keep]
    predict_cnt = len(points)
    
    # Desenhar as predições
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for i, p in enumerate(points):
        # Usar cores diferentes baseadas na confiança
        confidence = scores[keep][i]
        color = (0, int(255 * (1 - confidence)), int(255 * confidence))  # Verde para baixa confiança, vermelho para alta
        cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 3, color, -1)
        # Opcional: mostrar valor de confiança
        cv2.putText(img_to_draw, f'{confidence:.2f}', (int(p[0]), int(p[1])-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Salvar imagem com as detecções
    output_filename = os.path.join(output_dir, f'pred_{os.path.basename(img_path)}')
    cv2.imwrite(output_filename, img_to_draw)
    
    return predict_cnt

def nms(points, scores, threshold=0.3):
    """Non-Maximum Suppression para pontos"""
    if len(points) == 0:
        return []
    
    # Computar distâncias entre todos os pontos
    x = points[:, 0]
    y = points[:, 1]
    areas = np.ones_like(scores)
    
    # Ordenar por score
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Computar distâncias com os pontos restantes
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i], x[order[1:]])
        yy2 = np.minimum(y[i], y[order[1:]])
        
        w = np.maximum(0.0, xx1 - xx2)
        h = np.maximum(0.0, yy1 - yy2)
        
        # Calcular distância euclidiana
        dist = np.sqrt(w * w + h * h)
        
        # Manter apenas pontos distantes o suficiente
        inds = np.where(dist > threshold)[0]
        order = order[inds + 1]
    
    return keep

def main(args):
    # Configurar GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda')
    
    # Carregar modelo
    model = build_model(args)
    model.to(device)
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Transformação para pré-processamento
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Criar diretório de saída se não existir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Processar todas as imagens e gerar relatório
    results = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for filename in sorted(os.listdir(args.input_dir)):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(args.input_dir, filename)
            count = process_image(model, transform, img_path, device, args.output_dir,
                                confidence_threshold=args.confidence_threshold,
                                multi_scale=args.multi_scale,
                                scale_factors=args.scale_factors)
            results.append((filename, count))
    
    # Gerar relatório
    report_path = os.path.join(args.output_dir, 'detection_report.txt')
    with open(report_path, 'w') as f:
        f.write("Relatório de Detecção de Pessoas\n")
        f.write("================================\n\n")
        for filename, count in results:
            f.write(f"Imagem: {filename}\n")
            f.write(f"Pessoas detectadas: {count}\n")
            f.write("-" * 40 + "\n")
        
        total_people = sum(count for _, count in results)
        f.write(f"\nTotal de imagens processadas: {len(results)}\n")
        f.write(f"Total de pessoas detectadas: {total_people}\n")
        f.write(f"Média de pessoas por imagem: {total_people/len(results):.2f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet batch evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args) 