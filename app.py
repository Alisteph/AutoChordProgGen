import streamlit as st
import chord_gen


st.title('AutoChordProgressGenerater')


init_chords = st.text_input("input")
exec_generate = st.button('exec')

if exec_generate:
    output_chord = chord_gen.generate_chord(init_chords.split())

    output = st.text(''.join(output_chord))
    st.download_button(
        'download',
        open('tmp/chord6o.mid', 'br')
    )
