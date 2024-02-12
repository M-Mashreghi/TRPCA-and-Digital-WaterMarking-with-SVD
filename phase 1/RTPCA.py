!pip install tensorly
!pip install sporco
import os
import cv2
import numpy as np
import tifffile
import torch
from tqdm import tqdm
from sporco import cupy
if cupy.have_cupy:
    from sporco.cupy.admm.rpca import RobustPCA as RPCA_sporco_gpu
from sporco.admm.rpca import RobustPCA as RPCA_sporco
from torch.nn import functional as F
import tensorly as tl
import numpy as np
import torch
from tensorly import backend as T
from tensorly.base import fold, unfold
from tensorly.tenalg.proximal import soft_thresholding, svd_thresholding
import torch
import matplotlib.pyplot as plt



class RTPCA:
    def __init__(self,
                    video_path,
                    mask=None,
                    tol=10e-7,
                    reg_E=1,
                    reg_J=1,
                    mu_init=10e-5,
                    mu_max=10e9,
                    learning_rate=1.1,
                    n_iter_max=200,
                    verbose=1,
                    rescale_min = 0,
                    rescale_max = 255,
                    out_type = np.uint8,
                    backend='numpy'):
        self.video_path =video_path
        self.mask = mask
        self.Reconstruction_Error =list()
        self.tol = tol
        self.reg_E = reg_E
        self.reg_J = reg_J
        self.mu_init = mu_init
        self.mu_max = mu_max
        self.lr = learning_rate
        self.n_iter_max = n_iter_max
        self.verbose = verbose
        self.rescale_min= rescale_min
        self.rescale_max= rescale_max
        self.out_type= out_type
        self.backend = backend
        tl.set_backend(backend)
    def robust_pca(self , X, mask=None, tol=10e-7, reg_E=1, reg_J=1,
                    mu_init=10e-5, mu_max=10e9, learning_rate=1.1,
                     n_iter_max=100, verbose=1):
        if mask is None:
            mask = 1

        # Initialise the decompositions
        D = T.zeros_like(X, **T.context(X))  # low rank part

        E = T.zeros_like(X, **T.context(X))  # sparse part
        L_x = T.zeros_like(X, **T.context(X))  # Lagrangian variables for the (X - D - E - L_x/mu) term
        J = [T.zeros_like(X, **T.context(X)) for _ in range(T.ndim(X))] # Low-rank modes of X
        L = [T.zeros_like(X, **T.context(X)) for _ in range(T.ndim(X))] # Lagrangian or J
        # Norm of the reconstructions at each iteration
        rec_X = []

        rec_D = []
        mu = mu_init

        for iteration in range(n_iter_max):

            for i in range(T.ndim(X)):
                J[i] = fold(svd_thresholding(unfold(D, i) + unfold(L[i], i)/mu, reg_J/mu), i, X.shape)

            D = L_x/mu + X - E
            for i in range(T.ndim(X)):

                D += J[i] - L[i]/mu
            D /= (T.ndim(X) + 1)
            E = soft_thresholding(X - D + L_x/mu, mask*reg_E/mu)

            # Update the lagrangian multipliers
            for i in range(T.ndim(X)):

                L[i] += mu * (D - J[i])
                
            L_x += mu*(X - D - E)
            
            mu = min(mu*learning_rate, mu_max)

            # Evolution of the reconstruction errors
            rec_X.append(T.norm(X - D - E, 2))

            rec_D.append(max([T.norm(low_rank - D, 2) for low_rank in J]))
            # Convergence check
            # if iteration > 1:
            self.Reconstruction_Error.append(max(rec_X[-1], rec_D[-1]).item())
            if max(rec_X[-1], rec_D[-1]) <= tol:
                if verbose:
                    print('\nConverged in {} iterations'.format(iteration))
                break
            else:
                print("[INFO] iter:", iteration, " error:", (max(rec_X[-1], rec_D[-1]).item()))

        return D, E

    def separater(self, X):
        assert isinstance(X, torch.Tensor)
        X = torch.transpose(X, 0, 1)
        L, S = self.robust_pca(X, self.mask, self.tol, self.reg_E, self.reg_J,
                            self.mu_init, self.mu_max, self.lr,
                            self.n_iter_max, self.verbose)
        S = torch.transpose(S, 0, 1)
        L = torch.transpose(L, 0, 1)
        return L, S


    def normalize_rescale(self, x):
        normalized_x = np.divide(np.subtract(x, np.min(x)), np.ptp(x))

        if self.rescale_min == 0:  # in most cases
            rescaled_x = normalized_x * self.rescale_max
        else:
            rescaled_x = normalized_x * (self.rescale_max - self.rescale_min) + self.rescale_min

        return rescaled_x

    # get video info
    def video_info(self,capture):
            self.fps = int(capture.get(cv2.CAP_PROP_FPS))
            self.video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))



    def Make_Dict_clip(self, video_path):
        clip_dict = {'grayscale_frames': []}

        capture = cv2.VideoCapture()
        if not capture.open(video_path):
            raise ValueError(f'Unable to open: {video_path}')
        self.video_info(capture)
        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                # Assuming you want to store grayscale frames
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clip_dict['grayscale_frames'].append(gray_frame)
        finally:
            capture.release()

        return clip_dict



    def background_substraction(self, X):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_tensor = torch.from_numpy(X).to(device)

            bg_tensor, fg_tensor = self.separater(X_tensor)

            # Convert tensors to numpy arrays
            bg = bg_tensor.cpu().numpy()
            fg = fg_tensor.cpu().numpy()

            return fg, bg

        except Exception as e:
            print(f"Error during background subtraction: {e}")
            # Handle the error gracefully, e.g., return X as is or raise an exception
            return X, None

       
    def run(self):
         
        # read in video and process
        clip_dict = self.Make_Dict_clip(self.video_path)

        frames = clip_dict['grayscale_frames']
        frame_width = self.frame_width
        frame_height = self.frame_height
        video_length = self.video_length

        X = np.stack(frames, axis=-1)  # F,N
 
        X = X.reshape((-1, X.shape[-1]))
        x_std = np.std(X)
        x_mean = np.mean(X)
        # background_substraction
        X = (X-x_mean)/x_std
        # rearrange back to image.
        fg, bg = self.background_substraction(X)
        print("bg.shape:", bg.shape)
        print("fg.shape:", fg.shape)
        print("X.shape:", X.shape)
        self.plot_Reconstruction_Error()

        assert bg.shape == fg.shape == X.shape
        bg = np.transpose(bg)  # N,F
        fg = np.transpose(fg)  # N,F        
        bg = bg*x_std+x_mean        
        # min max       
        fg = fg*x_std+x_mean        
        bg = self.normalize_rescale(bg)      
        fg = self.normalize_rescale(fg)      
        fg = fg.reshape(video_length, frame_height, frame_width)
        bg = bg.reshape(video_length, frame_height, frame_width)

        output_path_background = "/content/output_video_background.mp4"
        output_path_foreground = "/content/output_video_foreground.mp4"
        self.create_video(bg, output_path_background)
        self.create_video(fg, output_path_foreground)



    
    def create_video(self, frames_data, output_path, fps=20):
        # Check if the output directory exists, create it if not
        output_directory = os.path.dirname(output_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Assuming all frames have the same dimensions
        frame_height, frame_width = frames_data.shape[1:3]

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

        # Write each frame to the video
        for frame_data in frames_data:
            # Ensure the frame data is a numpy array
            frame = np.asarray(frame_data, dtype=np.uint8)

            # Write the frame to the video
            video_writer.write(frame)

        # Release the VideoWriter object
        video_writer.release()
    def plot_Reconstruction_Error(self):
      # Create a line plot with multiple datasets
      plt.plot(self.Reconstruction_Error)

      # Add labels and legend
      plt.xlabel('Iter')
      plt.ylabel('Error')
      plt.title('Reconstruction_Error')
      plt.legend()

      # Display the plot
      plt.show()
    
        