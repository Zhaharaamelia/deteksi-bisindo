import numpy as np

def normalize_landmarks(landmarks):
    """
    Normalisasi 1 tangan (21 landmark)
    """
    if len(landmarks) == 0:
        return np.zeros((21, 3))

    landmarks = np.array(landmarks)

    base_x, base_y, base_z = landmarks[0]

    landmarks[:, 0] -= base_x
    landmarks[:, 1] -= base_y
    landmarks[:, 2] -= base_z

    return landmarks


def normalize_two_hands(landmarks):
    """
    Normalisasi khusus 2 tangan (42 landmark)

    Index 0-20  = tangan kiri
    Index 21-41 = tangan kanan
    """

    landmarks = np.array(landmarks)
    result = np.zeros((42, 3))

    # Tangan kiri
    if np.any(landmarks[0:21]):
        left_hand = landmarks[0:21].copy()

        left_base_x = left_hand[0][0]
        left_base_y = left_hand[0][1]
        left_base_z = left_hand[0][2]

        left_hand[:, 0] -= left_base_x
        left_hand[:, 1] -= left_base_y
        left_hand[:, 2] -= left_base_z

        result[0:21] = left_hand

    # Tangan kanan
    if np.any(landmarks[21:42]):
        right_hand = landmarks[21:42].copy()

        right_base_x = right_hand[0][0]
        right_base_y = right_hand[0][1]
        right_base_z = right_hand[0][2]

        right_hand[:, 0] -= right_base_x
        right_hand[:, 1] -= right_base_y
        right_hand[:, 2] -= right_base_z

        result[21:42] = right_hand

    return result


def extract_two_hands(multi_hand_landmarks, multi_handedness):
    """
    Mengambil maksimal 2 tangan.
    Index 0-20  = Left
    Index 21-41 = Right
    """

    combined_landmarks = np.zeros((42, 3))

    if not multi_hand_landmarks or not multi_handedness:
        return combined_landmarks

    for hand_idx, hand_landmarks in enumerate(multi_hand_landmarks):

        if hand_idx >= 2:
            break

        hand_label = multi_handedness[hand_idx].classification[0].label

        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ])

        norm_landmarks = normalize_landmarks(landmarks)

        if hand_label == "Left":
            combined_landmarks[0:21] = norm_landmarks

        elif hand_label == "Right":
            combined_landmarks[21:42] = norm_landmarks

    return combined_landmarks