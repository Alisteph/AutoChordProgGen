import streamlit as st
import chord_gen


st.title('AutoChordProgressGenerater')


init_chords = st.text_input("input")
exec = st.button('exec')

if exec:
    output_chord = chord_gen.generate_chord(init_chords.split())

    output = st.text(''.join(output_chord))
