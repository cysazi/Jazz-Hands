import time
import mido

outport = mido.open_output('My Virtual Port', virtual=True)

last_note = None

def get_glove_data():
    return Glove()

def position_to_note(position):
    y = position[0]

    min_y = - 0.5
    max_y = 0.5

    y_norm = (y - min_y) / (max_y - min_y)
    y_norm = max(0, min(1, y_norm))

    return int(60+ y_norm *24)

while True:

    glove = get_glove_data()

    note = position_to_note(glove.position)

    if note != last_note:
        # turn off previous note
        if last_note is not None:
            outport.send(mido.Message('note_off', note=last_note))

        # play new note
        outport.send(mido.Message('note_on', note=note, velocity=100))

        last_note = note

    time.sleep(0.02)  # ~20 ms latency