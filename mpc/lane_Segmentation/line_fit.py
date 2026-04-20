import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# feel free to adjust the parameters in the code if necessary


def viz1(binary_warped, ret, save_file=None):
	"""
	Visualize each sliding window location and predicted lane lines, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	out_img = ret['out_img']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	if save_file is None:
		plt.show()
	else:
		plt.savefig(save_file)
	plt.gcf().clear()


def bird_fit(binary_warped, ret, Minv):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']
	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = np.polyval(left_fit, ploty)
	right_fitx = np.polyval(right_fit, ploty)
	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 100  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))
	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 0, 255))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	result = cv2.warpPerspective(result, Minv, (1280, 720))
	return result


def final_viz(undist, left_fit, right_fit, m_inv):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Generate x and y values for plotting
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx = np.polyval(left_fit, ploty)
	right_fitx = np.polyval(right_fit, ploty)

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	# Convert arrays to 8 bit for later cv to ros image transfer
	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result


class Line():
	def __init__(self, n):
		"""
		n is the window size of the moving average
		"""
		self.n = n
		self.detected = False

		# Polynomial coefficients: x = A*y^2 + B*y + C
		# Each of A, B, C is a "list-queue" with max length n
		self.A = []
		self.B = []
		self.C = []
		# Average of above
		self.A_avg = 0.
		self.B_avg = 0.
		self.C_avg = 0.

	def get_fit(self):
		return (self.A_avg, self.B_avg, self.C_avg)

	def add_fit(self, fit_coeffs):
		"""
		Gets most recent line fit coefficients and updates internal smoothed coefficients
		fit_coeffs is a 3-element list of 2nd-order polynomial coefficients
		"""
		# Coefficient queue full?
		q_full = len(self.A) >= self.n

		# Append line fit coefficients
		self.A.append(fit_coeffs[0])
		self.B.append(fit_coeffs[1])
		self.C.append(fit_coeffs[2])

		# Pop from index 0 if full
		if q_full:
			_ = self.A.pop(0)
			_ = self.B.pop(0)
			_ = self.C.pop(0)

		# Simple average of line coefficients
		self.A_avg = np.mean(self.A)
		self.B_avg = np.mean(self.B)
		self.C_avg = np.mean(self.C)

		return (self.A_avg, self.B_avg, self.C_avg)


def lane_fit(binary_warped, nwindows=20, margin=50, minpix=10):
    """
    Finds lane lines using a sliding window approach with:
    1. Momentum-based tracking (handles gaps).
    2. Overlap prevention (windows physically cannot cross).
    3. Robust error handling.
    """    
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = int(binary_warped.shape[0] / nwindows)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []

    left_x_current = leftx_base
    right_x_current = rightx_base
    
    left_momentum = 0
    right_momentum = 0

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        
        separation = right_x_current - left_x_current
        
        if separation < (margin * 2):
            mid = (left_x_current + right_x_current) / 2
            left_x_current = int(mid - margin)
            right_x_current = int(mid + margin)

        win_xleft_low = left_x_current - margin
        win_xleft_high = left_x_current + margin
        win_xright_low = right_x_current - margin
        win_xright_high = right_x_current + margin

        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high) & 
                          (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox <= win_xright_high) & 
                           (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)).nonzero()[0]
        
        if len(good_left_inds) > minpix:
            new_x = int(np.mean(nonzerox[good_left_inds]))
            left_momentum = int(0.6 * (new_x - left_x_current) + 0.4 * left_momentum)
            left_x_current = new_x
            left_lane_inds.append(good_left_inds)
        else:
            left_x_current += left_momentum

        if len(good_right_inds) > minpix:
            new_x = int(np.mean(nonzerox[good_right_inds]))
            right_momentum = int(0.6 * (new_x - right_x_current) + 0.4 * right_momentum)
            right_x_current = new_x
            right_lane_inds.append(good_right_inds)
        else:
            right_x_current += right_momentum

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        print("Error: No lane pixels found.")
        return None

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    def fit_poly_safe(y, x):
        if len(y) < 50: return None 
        try:
            return np.polyfit(y, x, 2)
        except Exception:
            return None

    left_fit = fit_poly_safe(lefty, leftx)
    right_fit = fit_poly_safe(righty, rightx)
    
    if left_fit is None or right_fit is None:
        print("Error: Unable to fit polynomial to lanes.")
        return None

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    try:
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)
    except TypeError:
        return None


    ret = {
        'left_fit': left_fit,
        'right_fit': right_fit,
        'left_fitx': left_fitx,
        'right_fitx': right_fitx,
        'ploty': ploty,
        'nonzerox': nonzerox,
        'nonzeroy': nonzeroy,
        'left_lane_inds': left_lane_inds,
        'right_lane_inds': right_lane_inds
    }

    return ret

def perspective_transform(img, src):
    """
    Get bird's eye view from input image
    """
    height, width = img.shape[:2]
    dst = np.float32([(0,0), (0,height), (width, height), (width, 0)])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = np.linalg.inv(M)
    warped_img = cv2.warpPerspective(img, M, (width, height))
    return warped_img, M, Minv


def closest_point_on_polynomial(point, coeffs):
    """
    Finds the closest point on a polynomial curve x = P(y) to a target point.
    
    Args:
        point: tuple or list (x0, y0)
        coeffs: list of coefficients for P(y) [highest order first]
                e.g., [1, 0, 0] is x = y^2
    """
    x0, y0 = point

    P = np.poly1d(coeffs)
    P_deriv = P.deriv()
    
    y_poly = np.poly1d([1, 0])
    
    term1 = (P - x0) * P_deriv
    
    term2 = y_poly - y0
    
    dist_deriv_poly = term1 + term2
    
    roots = dist_deriv_poly.roots
    
    real_roots = roots[np.isreal(roots)].real
    
    min_dist_sq = float('inf')
    closest_x = None
    closest_y = None
    
    for y in real_roots:
        x = P(y)
        dist_sq = (x - x0)**2 + (y - y0)**2
        
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_x = x
            closest_y = y
            
    return np.array([closest_x, closest_y])