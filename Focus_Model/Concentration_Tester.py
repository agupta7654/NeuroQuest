import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from antropy import higuchi_fd

threshold_ch7 = None
threshold_ch8 = None

def get_resting_state(board):
    global threshold_ch7, threshold_ch8

    print("\n*** Resting State Calibration ***")
    print("Please relax, stay still, and breathe slowly for 30 seconds.")
    print("Capturing baseline brain activity...\n")

    # Collect 30 seconds of data
    rest_duration = 15
    sampling_rate = BoardShim.get_sampling_rate(board.get_board_id())
    total_samples_needed = sampling_rate * rest_duration

    all_data = []

    start_time = time.time()
    while time.time() - start_time < rest_duration:
        chunk = board.get_board_data()  # get all new samples so far
        if chunk.size > 0:
            all_data.append(chunk)
        time.sleep(0.2)

    # Combine all captured data
    if len(all_data) == 0:
        raise RuntimeError("No data captured during resting state.")

    data = np.hstack(all_data)

    # Compute Higuchi FD for channels 7 & 8
    threshold_ch7 = higuchi_fd(data[6])
    threshold_ch8 = higuchi_fd(data[7])

    print("Resting state FD values captured!")
    print(f"  Threshold CH7 = {threshold_ch7:.4f}")
    print(f"  Threshold CH8 = {threshold_ch8:.4f}\n")



# -------------------------------------
def check_concentration(board):
    global threshold_ch7, threshold_ch8

    print("\n*** Checking Concentration ***")

    while True:
        time.sleep(2)  # sample every 2 seconds

        data = board.get_current_board_data(1250)
        if data.shape[1] < 1000:  # not enough data yet
            continue
        if data.shape[1] > 50:
                data = data[:, 50:]  # remove first 50 samples if needed

        fd7 = higuchi_fd(data[6])
        fd8 = higuchi_fd(data[7])

        print(f"\nFD7 = {fd7:.4f}, FD8 = {fd8:.4f}")

        cond7 = fd7 > threshold_ch7
        cond8 = fd8 > threshold_ch8

        # Logic
        if cond7 and cond8:
            print(" → Concentrated")
        elif not cond7 and not cond8:
            print(" → Not Concentrated")
        else:
            print(" → Inconclusive")



# -------------------------------------
def main():
    global threshold_ch7, threshold_ch8

    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = "COM3"
    board_id = BoardIds.CYTON_DAISY_BOARD.value

    try:
        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()

        threshold_ch7 = None
        threshold_ch8 = None

        # 1. Calibrate
        get_resting_state(board)

        # 2. Run concentration loop
        check_concentration(board)

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        if 'board' in locals() and board.is_prepared():
            board.stop_stream()
            board.release_session()



if __name__ == "__main__":
    main()