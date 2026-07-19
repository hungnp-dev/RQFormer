from pathlib import Path
p = Path(r'H:\Master\Bao\code\RQFormer-2\_unicode_test.txt')
p.write_text('Cấu Hình Chạy - Tiếng Việt có dấu', encoding='utf-8')
print(p.read_text(encoding='utf-8'))
