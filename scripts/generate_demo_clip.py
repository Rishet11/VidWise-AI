import os
import sys
import subprocess
import tempfile
import shutil

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: PIL (Pillow) is required to run this script. Please install it with 'pip install Pillow'.")
    sys.exit(1)

def find_font():
    # Common font paths on macOS
    macos_fonts = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Cache/Helvetica.ttc"
    ]
    for path in macos_fonts:
        if os.path.exists(path):
            return path
    return None

def create_frames(temp_dir, width=1280, height=720):
    font_path = find_font()
    
    # Define steps for the 5-step shot list
    steps = [
        {
            "step": "1 / 5",
            "title": "Paste 3 YouTube Links",
            "desc": "Input up to six YouTube URLs or load a public playlist.",
            "color": "#1E1E2E", # Dark background
            "accent": "#3B82F6", # Blue
            "mockup": "URLs: \n- https://www.youtube.com/watch?v=dQw4w9WgXcQ\n- https://www.youtube.com/watch?v=kJQP7kiw5Fk\n- https://www.youtube.com/watch?v=9bZkp7q19f0",
            "duration": 3 # seconds
        },
        {
            "step": "2 / 5",
            "title": "Type Cross-Video Question",
            "desc": "Ask a technical or research question spanning all videos.",
            "color": "#1E1E2E",
            "accent": "#A855F7", # Purple
            "mockup": "Question: 'What are the main arguments about remote work across these talks?'",
            "duration": 3
        },
        {
            "step": "3 / 5",
            "title": "Show Answer with Citation Chips",
            "desc": "VidWise generates an answer. Every claim gets a second-accurate citation chip.",
            "color": "#1E1E2E",
            "accent": "#10B981", # Teal/Green
            "mockup": "Answer: 'Remote work increases productivity [Video 1 @ 02:15] but reduces collaboration [Video 2 @ 10:45].'",
            "duration": 3
        },
        {
            "step": "4 / 5",
            "title": "Expand Evidence Panel",
            "desc": "Click a citation chip to view the exact transcript snippet and source context.",
            "color": "#1E1E2E",
            "accent": "#F59E0B", # Orange
            "mockup": "[Evidence Panel]\nVideo 2 (10:45): '...when people are fully remote, spontaneous collaboration drops by 30%...'",
            "duration": 3
        },
        {
            "step": "5 / 5",
            "title": "Click Citation to Jump to YouTube",
            "desc": "Click the citation chip to open YouTube at the exact timestamp (e.g. &t=645s).",
            "color": "#1E1E2E",
            "accent": "#EF4444", # Red
            "mockup": "Redirecting to: youtube.com/watch?v=kJQP7kiw5Fk&t=645s",
            "duration": 3
        }
    ]
    
    frame_idx = 0
    for i, step in enumerate(steps):
        # Create an image frame
        img = Image.new("RGB", (width, height), color=step["color"])
        draw = ImageDraw.Draw(img)
        
        # Load fonts if available, otherwise use default
        try:
            if font_path:
                title_font = ImageFont.truetype(font_path, 48)
                desc_font = ImageFont.truetype(font_path, 24)
                mockup_font = ImageFont.truetype(font_path, 20)
                step_font = ImageFont.truetype(font_path, 32)
                footer_font = ImageFont.truetype(font_path, 16)
            else:
                raise IOError
        except IOError:
            title_font = desc_font = mockup_font = step_font = footer_font = ImageFont.load_default()
            
        # Draw top header/logo
        draw.text((50, 40), "VidWise Demo", fill="#94A3B8", font=footer_font)
        
        # Draw step index
        draw.rectangle([50, 90, 150, 130], fill=step["accent"])
        draw.text((70, 95), step["step"], fill="#FFFFFF", font=step_font)
        
        # Draw title
        draw.text((50, 160), step["title"], fill="#FFFFFF", font=title_font)
        
        # Draw description
        draw.text((50, 240), step["desc"], fill="#94A3B8", font=desc_font)
        
        # Draw a visual mockup box
        draw.rectangle([50, 310, width - 50, height - 100], outline=step["accent"], width=2)
        
        # Draw mockup text inside the box
        draw.text((75, 335), step["mockup"], fill="#E2E8F0", font=mockup_font)
        
        # Draw footer progress bar
        bar_y = height - 40
        draw.rectangle([50, bar_y, width - 50, bar_y + 8], fill="#334155")
        # Draw active part of progress bar
        progress_width = int((i + 1) / len(steps) * (width - 100))
        draw.rectangle([50, bar_y, 50 + progress_width, bar_y + 8], fill=step["accent"])
        
        # Write frames for each second of duration
        for _ in range(step["duration"]):
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            img.save(frame_path)
            frame_idx += 1
            
    return frame_idx

def generate_video():
    temp_dir = tempfile.mkdtemp()
    print(f"Generating temporary frames in {temp_dir}...")
    
    num_frames = create_frames(temp_dir)
    output_path = "/Users/rishetmehra/Desktop/vidwise-ai/docs/demo_clip.mp4"
    
    # Ensure docs directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Attempt compilation using cv2 (OpenCV) if available
    try:
        import cv2
        import numpy as np
        print("OpenCV found. Attempting to write video using cv2.VideoWriter...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # fallback MP4 codec
        # Try H.264 standard codec if possible
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except:
            pass
            
        video = cv2.VideoWriter(output_path, fourcc, 1, (1280, 720))
        for idx in range(num_frames):
            frame_file = os.path.join(temp_dir, f"frame_{idx:04d}.png")
            frame_img = cv2.imread(frame_file)
            video.write(frame_img)
        video.release()
        print(f"Successfully generated video at {output_path} using OpenCV.")
        shutil.rmtree(temp_dir)
        return True
    except ImportError:
        print("OpenCV not found in Python environment.")
        
    # Attempt compilation using ffmpeg via subprocess
    print("Checking if ffmpeg is available in system PATH...")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print("ffmpeg found. Running subprocess to stitch frames...")
        # Stitch frames into 1fps video, output standard libx264 MP4
        cmd = [
            "ffmpeg", "-y",
            "-framerate", "1",
            "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully generated video at {output_path} using ffmpeg.")
            shutil.rmtree(temp_dir)
            return True
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed with exit code {e.returncode}")
    else:
        print("ffmpeg command not found in system PATH.")
        
    shutil.rmtree(temp_dir)
    print("Error: Could not generate MP4 video. Both OpenCV and ffmpeg were unavailable or failed.")
    return False

if __name__ == "__main__":
    success = generate_video()
    if not success:
        sys.exit(1)
