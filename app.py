from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO
import base64
import re
import traceback

app = Flask(__name__)
CORS(app)

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

출력 형식:
```python
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_aspect('equal')
ax.axis('off')

# 도형 그리기 코드...

fig.tight_layout()
fig  # 반환
```"""

def extract_python_code(text):
    """응답에서 Python 코드 추출"""
    # ```python ... ``` 블록 찾기
    match = re.search(r'```python\s*([\s\S]*?)```', text)
    if match:
        return match.group(1).strip()
    # ``` ... ``` 블록 찾기
    match = re.search(r'```\s*([\s\S]*?)```', text)
    if match:
        return match.group(1).strip()
    # 코드 블록 없으면 전체 텍스트 반환
    return text.strip()

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
        api_key = data.get('anthropic_key')

        if not image_base64:
            return jsonify({'error': 'image is required'}), 400
        if not api_key:
            return jsonify({'error': 'anthropic_key is required'}), 400

        # 치수 라벨 교체 지시 생성
        changes_instruction = ""
        if changes:
            changes_list = ', '.join([f'"{c["original"]}" → "{c["new"]}"' for c in changes])
            changes_instruction = f"\n\n치수 라벨을 다음과 같이 교체해서 그려주세요: {changes_list}"

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
            model="claude-sonnet-4-20250514",
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
                        "text": f"이 수학 도형을 matplotlib으로 정확히 재현하는 Python 코드를 생성해주세요.{changes_instruction}"
                    }
                ]
            }]
        )

        # 응답에서 코드 추출
        response_text = message.content[0].text
        code = extract_python_code(response_text)

        print(f"[draw] Generated code:\n{code[:500]}...")

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
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
