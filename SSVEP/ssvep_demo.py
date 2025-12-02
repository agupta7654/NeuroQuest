import pygame
import math
import socket
import threading
import time

# --- CONFIGURATION ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)

# Circle Settings
CIRCLE_RADIUS = 80
LEFT_CIRCLE_POS = (200, 300)
RIGHT_CIRCLE_POS = (600, 300)
LEFT_FREQ = 8.0   # Hz
RIGHT_FREQ = 14.0 # Hz

# UDP Settings (Listening for Model)
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# --- GLOBAL STATE ---
detected_text = "Waiting for Input... (Press A or B to test)"
last_detection_time = 0
simulation_mode = False # Set to False when using real headset

def udp_listener():
    """Listens for real BCI commands over UDP."""
    global detected_text, last_detection_time
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        sock.bind((UDP_IP, UDP_PORT))
        print(f"Listening for model commands on {UDP_IP}:{UDP_PORT}")
        
        while True:
            data, addr = sock.recvfrom(1024)
            message = data.decode("utf-8")
            process_command(message)
            
    except OSError as e:
        print(f"UDP Listener Error: {e}")


def process_command(command):
    """Updates the UI state based on the received command."""
    global detected_text, last_detection_time
    
    if command == "INPUT_8HZ":
        detected_text = "DETECTED: 8 Hz (Left)"
        last_detection_time = time.time()
    elif command == "INPUT_14HZ":
        detected_text = "DETECTED: 14 Hz (Right)"
        last_detection_time = time.time()

# --- MAIN GAME LOOP ---
def main():
    global detected_text, last_detection_time
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("NeuroQuest SSVEP Test (Press A/B to Simulate)")
    font = pygame.font.Font(None, 48)
    clock = pygame.time.Clock()
    
    # Start UDP thread (it runs in background even if we simulate)
    thread = threading.Thread(target=udp_listener, daemon=True)
    thread.start()
    
    running = True
    start_time = time.time()
    
    while running:
        # 1. Event Handling (Includes Simulation Keys)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # --- SIMULATION MODE: KEYBOARD INPUT ---
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:  # Press 'A' to simulate 8Hz
                    process_command("INPUT_8HZ")
                elif event.key == pygame.K_b: # Press 'B' to simulate 14Hz
                    process_command("INPUT_14HZ")
        
        # 2. Clear Screen
        screen.fill(BG_COLOR)
        
        # 3. Draw Flashing Circles
        current_time = time.time() - start_time
        
        # Left Circle (8 Hz)
        # Sine wave: (sin(2*pi*f*t) + 1) / 2 maps -1..1 to 0..1
        intensity_left = (math.sin(2 * math.pi * LEFT_FREQ * current_time) + 1) / 2
        color_val_left = int(intensity_left * 255)
        pygame.draw.circle(screen, (color_val_left, color_val_left, color_val_left), LEFT_CIRCLE_POS, CIRCLE_RADIUS)
        
        # Right Circle (14 Hz)
        intensity_right = (math.sin(2 * math.pi * RIGHT_FREQ * current_time) + 1) / 2
        color_val_right = int(intensity_right * 255)
        pygame.draw.circle(screen, (color_val_right, color_val_right, color_val_right), RIGHT_CIRCLE_POS, CIRCLE_RADIUS)
        
        # 4. Draw Labels
        label_left = font.render("8 Hz", True, (100, 100, 100))
        label_right = font.render("14 Hz", True, (100, 100, 100))
        screen.blit(label_left, (LEFT_CIRCLE_POS[0]-30, LEFT_CIRCLE_POS[1]+100))
        screen.blit(label_right, (RIGHT_CIRCLE_POS[0]-40, RIGHT_CIRCLE_POS[1]+100))

        # 5. Draw Detection Text
        # Reset text after 1 second if no new signal comes in
        if time.time() - last_detection_time > 1.0:
             detected_text = "Waiting..."
             text_color = (100, 100, 100)
        else:
             text_color = (0, 255, 0) # Green for active detection

        text_surface = font.render(detected_text, True, text_color)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH/2, 100))
        screen.blit(text_surface, text_rect)
        
        pygame.display.flip()
        clock.tick(60) # Run at 60 FPS

    pygame.quit()

if __name__ == "__main__":
    main()