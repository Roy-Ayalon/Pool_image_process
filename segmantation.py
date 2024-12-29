import cv2
import numpy as np
import os

def draw_label(img, text, center, color=(255, 0, 0), font_scale=0.6, thickness=2):
    """
    Draw text label near the given center coordinates on the image.
    """
    x, y = center
    # Use a simple sans-serif font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Put text slightly above the center
    cv2.putText(img, text, (x - 20, y - 10), font, font_scale, color, thickness, cv2.LINE_AA)



def get_board_color_rgb(color_img):
    """
    Given an image (as a NumPy array in BGR format), return the average board color in (R, G, B) format.

    :param color_img: Input image in BGR format (as read by OpenCV).
    :return: (R, G, B) tuple of integers representing the average color.
    """
    # find biggest rectangle in the image
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # show the image
    cv2.imshow("Gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # threshold the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # show the image
    cv2.imshow("Threshold", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #show the contours
    cv2.drawContours(color_img, contours, -1, (0, 0, 255), 2)
    cv2.imshow("Contours", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # find the biggest contour
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            biggest_contour = contour
    
    # find the bounding rectangle
    x, y, w, h = cv2.boundingRect(biggest_contour)
    roi = color_img[y:y+h, x:x+w]

    # draw the rectangle
    cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Rectangle", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print the dominant color
    dominant_color = roi.mean(axis=(0, 1))
    print(f"Dominant color of the board: R={dominant_color[2]}, G={dominant_color[1]}, B={dominant_color[0]}")
    board_color = (dominant_color[2], dominant_color[1], dominant_color[0])

    return board_color
    


def find_white_ball(color_img):
    """
    Find the white ball in the image by:
      1) Converting to HSV color space
      2) Thresholding for white color
      3) Applying morphological operations to clean up the mask
      4) Finding contours and approximating the largest one as a circle
      5) Drawing the circle and label on the image
    """
    # Convert to HSV
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

    #show all the channels
    h, s, v = cv2.split(hsv)
    cv2.imshow("Hue", h)
    cv2.imshow("Saturation", s)
    cv2.imshow("Value", v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # take s channel, and threshold it
    _, s_thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("Saturation Threshold", s_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    #find black circle
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([255, 10, 255])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    cv2.imshow("Black Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Define white color range
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 10, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No white ball found. Try adjusting threshold or lighting.")
        return color_img, None
    
    # Draw all contours in green
    cv2.drawContours(color_img, contours, -1, (0, 255, 0), 2)

    #show the image
    cv2.imshow("Contours", color_img)
    
    # Filter contours by area (minimum size threshold)
    min_area = 500
    max_area = 700
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    if not filtered_contours:
        print("No valid contours found after filtering.")
        return color_img, None
    
    # Find the largest contour
    largest_contour = max(filtered_contours, key=cv2.contourArea)
    
    # Approximate the largest contour as a circle
    ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
    if radius < 5:  # Validate the radius
        print("Detected circle is too small.")
        return color_img, None
    
    center = (int(x), int(y))
    radius = int(radius)
    
    # Draw the circle on the image
    cv2.circle(color_img, center, radius, (255, 0, 0), 2)
    draw_label(color_img, 'White Ball', (center[0] + 10, center[1] - 10))
    
    return color_img, largest_contour

def detect_green_table_edges(color_img):
    """
    Detects the edges of a green pool table by using the 'a' channel of the Lab color space:
      1) Convert to Lab color space and split L, a, b.
      2) Threshold the a-channel to isolate green.
      3) Find the largest contour.
      4) Approximate that contour as a polygon.
      5) Draw the polygon (and corners) on the image.
      
    :param color_img: BGR image (numpy array).
    :return: (annotated_image, polygon_points)
    """
    color_img_copy = color_img.copy()
    #first, find white ball and remove it
    color_img, white_ball_contour = find_white_ball(color_img)
    board_color = get_board_color_rgb(color_img_copy)




    # remove the white ball from the image, by filling the contour with green as background
    cv2.drawContours(color_img, [white_ball_contour], -1, board_color, -1)
   


    #show the image
    cv2.imshow("No White Ball", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Convert to Lab color space
    lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # -- Debug/Visualization (optional) --
    # cv2.imshow("L Channel", L)
    # cv2.imshow("A Channel", A)
    # cv2.imshow("B Channel", B)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Typically, in Lab space, 'A' ranges around 128 for neutral gray,
    # values < 128 are more greenish, > 128 are more reddish.
    # We'll do a simple threshold to find "green" region.
    # You may need to adjust these bounds based on your table/felt.
    
    # Approach 1: Otsu threshold on the A channel
    # This attempts to automatically find a suitable threshold.
    # Then invert the threshold if needed to get the table as white on black background.
    _, thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Threshold for green in the 'A' channel
    lower_green = 0    # Adjust this based on your image
    upper_green = 120  # Adjust this based on your image
    thresh = cv2.inRange(A, lower_green, upper_green)

    # Apply morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # show the thresholded image
    cv2.imshow("Thresholded Image", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    # If Otsu isn't working well, you can manually set bounds, e.g.:
    # thresh = cv2.inRange(A, 0, 120)  # This is an alternative approach.

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found for the green table. Try adjusting threshold or lighting.")
        return color_img, None
    
    # draw the contours
    cv2.drawContours(color_img, contours, -1, (0, 0, 255), 2)

    # show the image
    cv2.imshow("Detected Green Table", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # make the contours more smooth, in shape of rectangle
    for contour in contours:
        epsilon = 0.0002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(color_img, [approx], -1, (0, 255, 0), 3)

    # show the image
    cv2.imshow("Detected Green Table", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Pick the largest contour (most likely the table)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate that contour to a polygon
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Draw the polygon on the original color image
    cv2.drawContours(color_img, [approx_polygon], -1, (255, 0, 0), 3)

    # Optionally draw corners and label them
    for idx, corner in enumerate(approx_polygon):
        x, y = corner[0]
        cv2.circle(color_img, (x, y), 5, (255, 0, 0), -1)
        draw_label(color_img, f"Corner {idx+1}", (x, y))

    return color_img, approx_polygon

   

def detect_holes(gray_img, color_img):
    """
    Detect circular holes using HoughCircles.
    Returns the image with circles drawn and a list of centers.
    """
    # HoughCircles works better with blur
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30, 
        param1=100, 
        param2=30, 
        minRadius=10, 
        maxRadius=40
    )
    
    hole_centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i
            # Draw the circle in blue
            cv2.circle(color_img, (x, y), r, (255, 0, 0), 2)
            hole_centers.append((x, y, r))
            draw_label(color_img, 'Hole', (x, y))
    
    return color_img, hole_centers

def detect_balls(gray_img, color_img):
    """
    Detect circular balls using HoughCircles, then label each with a unique ID.
    Returns the image with circles drawn and a list of ball centers.
    """
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )
    
    ball_centers = []
    ball_id = 1
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i
            # Draw the circle in blue
            cv2.circle(color_img, (x, y), r, (255, 0, 0), 2)
            
            # Label the ball by ID
            draw_label(color_img, f"Ball #{ball_id}", (x, y))
            ball_centers.append((x, y, r, ball_id))
            ball_id += 1
    
    return color_img, ball_centers

def detect_cue_stick(color_img):
    """
    A naive approach to detecting the pool cue (stick) by:
      1) Converting to HSV
      2) Thresholding for a typical 'wood'/brown range (or whichever color you expect)
      3) Using HoughLinesP to find the elongated shape
    You will likely need to adapt the color range based on your actual cue stick.
    """
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    
    # Example “brown”-ish hue range; adjust for your image.
    lower_brown = np.array([5, 50, 50])
    upper_brown = np.array([20, 255, 255])
    
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Optionally do some morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=70, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw in blue
            cv2.line(color_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Label the stick - (this is naive, will label each line)
        # If your code finds multiple line segments, you might want to combine them or pick the longest
        draw_label(color_img, 'Stick', (lines[0][0][0], lines[0][0][1]))
    
    return color_img


def main():
    # Load your images
    for img in range(1, 4):
        image_path = f"first_pics/{img}.jpeg"
        color_img = cv2.imread(image_path)
        if color_img is None:
            print(f"Could not load image {image_path}.")
            continue
        
        # convert to lab
        lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # max of a channel, and min
        print(f"Max of a channel: {a.max()}")
        print(f"Min of a channel: {a.min()}")
        

        # Detect white ball in the image
        color_img, white_ball_contour = find_white_ball(color_img)
        
        # Show the image
        cv2.imshow("White Ball", color_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    

if __name__ == "__main__":
    main()