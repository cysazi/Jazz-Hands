from SoundEngine import SoundEngine, render_song, save_to_wav

if __name__ == "__main__":
    raw_data = """60,100,0.0,64,Piano,0
62,110,-0.2,70,Piano,1
64,120,0.2,80,Piano,0
65,90,-0.1,60,Guitar,1
67,105,0.1,75,Guitar,2
69,115,-0.3,85,Guitar,0"""

    engine = SoundEngine()

    print("Rendering audio...")
    audio = render_song(raw_data, engine)

    save_to_wav("output.wav", audio)

    engine.close()

    print("Saved to output.wav")