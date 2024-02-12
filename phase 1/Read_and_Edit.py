
!apt-get install ffmpeg

# Get video duration using FFprobe
import subprocess
import json
import torch
import torchvision.transforms as transforms
from google.colab.patches import cv2_imshow
import cv2
import numpy as np




class Read_and_Edit_Clip:
    def __init__(self,video_path):
        self.video_path = video_path
    def read_clip(self):
        # Open the video file
        self.cap = cv2.VideoCapture(self.video_path)
        # Select a frame number to display (for example, the 100th frame)
        self.frame_number_to_display = 120
        # Set the frame position to the selected frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number_to_display)
        # Read the selected frame
        ret, frame = self.cap.read()
        # Display the frame
        cv2_imshow( frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Display the resolution of the selected frame
        self.height, self.width, _ = frame.shape
        # Release the video capture object
        self.cap.release()
        # Convert the NumPy array to a PyTorch tensor
        self.frame_tensor = transforms.ToTensor()(frame)
        # Individual dimensions
        self.channels, self.height, self.width = self.frame_tensor.shape
    def show_information(self):
        print("Channels:", self.channels)
        print("Height:", self.height)
        print("Width:", self.width)
        print(f"Resolution of Frame {self.frame_number_to_display}: {self.width}xself.{self.height}")
        print("Tensor Shape (dimensions):", self.frame_tensor.shape)

    def change_greay_mode(self):
        # Convert the frame to black and white (grayscale)
        gray_frame = cv2.cvtColor( self.frame, cv2.COLOR_BGR2GRAY)
        # Convert the NumPy array to a PyTorch tensor
        frame_tensor = transforms.ToTensor()(gray_frame)
        # Display the black and white frame
        cv2_imshow( gray_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Release the video capture object
        self.cap.release()


    def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
        noisy_image = np.copy(image)
        total_pixels = image.size
    
        # Add salt noise
        num_salt = np.ceil(salt_prob * total_pixels)
        salt_coordinates = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[salt_coordinates[0], salt_coordinates[1]] = 1
    
        # Add pepper noise
        num_pepper = np.ceil(pepper_prob * total_pixels)
        pepper_coordinates = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[pepper_coordinates[0], pepper_coordinates[1]] = 0
    
        return noisy_image
    
    def add_noise(self):
        # Read the video
        cap = cv2.VideoCapture(self.video_path)
        # Check if the video file is opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        # Get video details
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(5)
        self.noise_clip = "output_video.avi"
        # Create VideoWriter object to save the output video
        out = cv2.VideoWriter(self.noise_clip, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Add salt-and-pepper noise to the grayscale frame
            noisy_gray_frame = self.add_salt_and_pepper_noise(gray_frame, salt_prob=0.01, pepper_prob=0.01)
            # Convert grayscale frame back to RGB
            noisy_frame = cv2.cvtColor(noisy_gray_frame, cv2.COLOR_GRAY2BGR)
            # Write the frame to the output video
            out.write(noisy_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        # Release the video capture and writer objects
        cap.release()
        out.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()

    def video_compressore(self):

        # Use FFmpeg to compress the video
        input_video_path = self.noise_clip
        self.compress_with_noise = 'output_video_compressed.avi'
        output_video_path = 'output_video_compressed.avi'

        !ffmpeg -i $input_video_path -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k $output_video_path

    def Split_Clip(self):
        
        input_video_path = self.compress_with_noise

        # Run FFprobe to get video information in JSON format
        ffprobe_cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams {input_video_path}'
        info_json = subprocess.check_output(ffprobe_cmd, shell=True).decode('utf-8')
        video_info = json.loads(info_json)

        # Extract video duration
        self.video_duration = float(video_info['format']['duration'])

        # Calculate split points based on duration
        split_point = self.video_duration / 2  # Split the video into two equal parts

        # Output video paths
        first_clip = 'output_video_part1.avi'
        second_clip = 'output_video_part2.avi'

        # Use FFmpeg to split the video into two parts
        !ffmpeg -i $input_video_path -t $split_point -c copy $first_clip
        !ffmpeg -i $input_video_path -ss $split_point -c copy $second_clip
        return first_clip,second_clip