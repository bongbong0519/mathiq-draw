from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO
import base64
import re
import traceback
import cv2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

SYSTEM_PROMPT = """당신은 수학 문제의 도형을 matplotlib 코드로 정밀하게 재현하는 전문가입니다.

규칙:
1. 주어진 이미지의 도형을 분석하고 matplotlib으로 동일하게 그리는 Python 코드를 생성하세요.
2. 반드시 실행 가능한 Python 코드만 출력하세요. 설명이나 마크다운 없이.
3. 코드는 fig, ax를 생성하고 도형을 그린 뒤 fig 객체를 반환해야 합니다.
4. 다음 변수들이 미리 정의되어 있다고 가정하세요: plt, patches, np
5. 코드 마지막에 반드시 fig를 반환하세요.

스타일 규칙:
- 배경: 흰색
- 선분: color='#1e293b', linewidth=1.5
- 도형 내부: facecolor='#e5e7eb', alpha=0.5
- 텍스트 라벨: fontsize=12, fontfamily='serif'
- 직각 기호: 8x8 크기의 작은 사각형
- 꼭짓점: 반지름 3의 점 (scatter)
- 축 숨기기: ax.axis('off')
- 여백 최소화: fig.tight_layout()

치수 라벨 교체가 요청된 경우, 해당 숫자들을 새 값으로 변경해서 그리세요.

도형 분석 순서:
1. 전체 직사각형의 가로(a), 세로(b) 파악
2. 정사각형 분할 위치 계산 (좌상단부터 시계방향)
3. 각 직각이등변삼각형의 직각 꼭짓점 위치 파악
4. 대각선 방향 정확히 재현
5. 치수선 위치(상단 가로, 우측 세로) 재현

특히 삼각형의 직각 위치와 대각선 방향을 원본과 동일하게 그릴 것.

출력 형식:
```python
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_aspect('equal')
ax.axis('off')

# 도형 그리기 코드...

fig.tight_layout()
fig  # 반환
```

중요: 응답은 반드시 ```python 코드블록 하나만 출력하세요.
코드블록 앞뒤에 설명, 분석, 주석을 절대 넣지 마세요.
분석이 필요하면 코드 안에 주석(#)으로 넣으세요."""

def extract_python_code(text):
    """응답에서 Python 코드 추출 - 가장 긴 코드블록 선택"""
    patterns = [
        r'```python\s*([\s\S]*?)```',
        r'```py\s*([\s\S]*?)```',
        r'```\s*([\s\S]*?)```',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            longest = max(matches, key=len)
            return longest.strip()
    # 코드블록 없으면 ``` 라인 제거 후 반환
    lines = text.strip().split('\n')
    code_lines = [l for l in lines if not l.startswith('```')]
    return '\n'.join(code_lines).strip()

def execute_matplotlib_code(code):
    """matplotlib 코드 실행하고 PNG base64 반환"""
    # 실행 환경 설정
    exec_globals = {
        'plt': plt,
        'patches': patches,
        'np': np,
        'matplotlib': matplotlib
    }
    exec_locals = {}

    # 코드 실행
    exec(code, exec_globals, exec_locals)

    # fig 객체 찾기
    fig = exec_locals.get('fig') or exec_globals.get('fig')
    if fig is None:
        raise ValueError("코드에서 fig 객체를 찾을 수 없습니다")

    # PNG로 저장
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)

    # base64 인코딩
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/draw', methods=['POST'])
def draw():
    try:
        data = request.json
        image_base64 = data.get('image')
        changes = data.get('changes', [])

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({'error': '서버에 ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다'}), 500

        if not image_base64:
            return jsonify({'error': 'image is required'}), 400

        # 치수 라벨 교체 지시 생성
        changes_instruction = ""
        if changes:
            changes_list = '\n'.join([f'  - "{c["original"]}" → "{c["new"]}"' for c in changes])
            changes_instruction = f"""

[필수 치수 교체 - 반드시 적용]
다음 값들을 코드에서 반드시 새 값으로 바꿔야 합니다:
{changes_list}

예: width=3 이면 width=4로, height=2+np.sqrt(7) 이면 height=3+np.sqrt(5)로 변경.
원본 값을 절대 사용하지 마세요."""

        # Claude API 호출
        client = anthropic.Anthropic(api_key=api_key)

        # 이미지 MIME 타입 추정
        if image_base64.startswith('/9j/'):
            media_type = 'image/jpeg'
        elif image_base64.startswith('iVBOR'):
            media_type = 'image/png'
        else:
            media_type = 'image/png'

        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": f"이 수학 도형을 matplotlib으로 재현하는 Python 코드를 생성해주세요.{changes_instruction}"
                    }
                ]
            }]
        )

        # 응답에서 코드 추출
        response_text = message.content[0].text
        code = extract_python_code(response_text)

        print(f"[draw] Generated code:\n{code}")

        # 코드 실행하여 이미지 생성
        result_base64 = execute_matplotlib_code(code)

        return jsonify({
            'success': True,
            'image': result_base64,
            'code': code
        })

    except anthropic.APIError as e:
        print(f"[draw] Anthropic API error: {e}")
        return jsonify({'error': f'API error: {str(e)}'}), 500
    except Exception as e:
        print(f"[draw] Error: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'code': code if 'code' in dir() else None}), 500

@app.route('/draw-tikz', methods=['POST'])
def draw_tikz():
    try:
        data = request.json
        tikz_code = data.get('tikz')
        changes = data.get('changes', [])

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({'error': '서버에 ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다'}), 500

        if not tikz_code:
            return jsonify({'error': 'tikz is required'}), 400

        # 치수 라벨 교체 지시 생성
        changes_instruction = ""
        if changes:
            changes_list = '\n'.join([f'  - "{c["original"]}" → "{c["new"]}"' for c in changes])
            changes_instruction = f"""

[필수 치수 교체 - 반드시 적용]
다음 값들을 코드에서 반드시 새 값으로 바꿔야 합니다:
{changes_list}

원본 값을 절대 사용하지 마세요."""

        # Claude API 호출
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"다음 TikZ 코드를 matplotlib Python 코드로 변환해줘.{changes_instruction}\n\nTikZ 코드:\n{tikz_code}"
            }]
        )

        # 응답에서 코드 추출
        response_text = message.content[0].text
        code = extract_python_code(response_text)

        print(f"[draw-tikz] Generated code:\n{code}")

        # 코드 실행하여 이미지 생성
        result_base64 = execute_matplotlib_code(code)

        return jsonify({
            'success': True,
            'image': result_base64,
            'code': code
        })

    except anthropic.APIError as e:
        print(f"[draw-tikz] Anthropic API error: {e}")
        return jsonify({'error': f'API error: {str(e)}'}), 500
    except Exception as e:
        print(f"[draw-tikz] Error: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'code': code if 'code' in dir() else None}), 500

@app.route('/inpaint', methods=['POST'])
def inpaint():
    """OpenCV 인페인팅으로 라벨 영역의 텍스트 제거"""
    try:
        data = request.json
        image_base64 = data.get('image')
        media_type = data.get('mediaType', 'image/png')
        labels = data.get('labels', [])
        padding = data.get('padding', 0.02)

        if not image_base64:
            return jsonify({'error': 'image is required'}), 400

        if not labels:
            # 라벨 없으면 원본 그대로 반환
            return jsonify({'success': True, 'image': image_base64})

        # base64 디코딩
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': '이미지 디코딩 실패'}), 400

        h, w = image.shape[:2]
        print(f"[inpaint] 이미지 크기: {w}x{h}, 라벨 수: {len(labels)}, padding: {padding}")

        # 마스크 생성 (검은색 배경)
        mask = np.zeros((h, w), dtype=np.uint8)

        # 각 라벨 영역을 흰색으로 마스킹
        for label in labels:
            # 정규화 좌표를 픽셀 좌표로 변환
            lx = label.get('x', 0)
            ly = label.get('y', 0)
            lw = label.get('w', 0)
            lh = label.get('h', 0)

            # 패딩 적용
            x1 = int(max(0, (lx - padding) * w))
            y1 = int(max(0, (ly - padding) * h))
            x2 = int(min(w, (lx + lw + padding) * w))
            y2 = int(min(h, (ly + lh + padding) * h))

            print(f"[inpaint] 라벨 영역: ({x1}, {y1}) ~ ({x2}, {y2})")

            # 마스크에 흰색 사각형 그리기
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # 인페인팅 실행
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

        # 결과를 PNG base64로 인코딩
        _, buffer = cv2.imencode('.png', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        print(f"[inpaint] 인페인팅 완료, 결과 크기: {len(result_base64)} bytes")

        return jsonify({
            'success': True,
            'image': result_base64
        })

    except Exception as e:
        print(f"[inpaint] Error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
