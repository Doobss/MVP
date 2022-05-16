import React, { useState, useContext, useEffect } from 'react';
import api from './api'
import styled from 'styled-components';
import { renderProtien, buildProtiens } from './methods'
import Renderer from './components/renderer'


export default function App() {
  const [dna, setDna] = useState([])
  const [legend, setLegend] = useState(false)
  const [chains, setChains] = useState([[]])
  const [colorBy, setColorBy] = useState('order')
  const [rendered, setRendered] = useState(0)
  const [protiens, setProtiens] = useState([[]])

  useEffect(() => {
    if (!protiens[0].vecs) {

      api.get.create()
        .then(created => {
          const createdProtiens = buildProtiens(created)
          setProtiens(createdProtiens)
          renderProtien(createdProtiens[rendered], colorBy, legend)
        })

    }
  }, [protiens, setProtiens])

  useEffect(() => {
    renderProtien(protiens[rendered], colorBy, legend)
  }, [rendered, colorBy, legend])



  return (
    <RendererSection id="renderer" >
      <Title>Protein: {rendered + 1}  Amino acid count {protiens[rendered].vecs && protiens[rendered].vecs.length}</Title>
      <Footer>
        {rendered ?
          <ScrollButton
            onClick={() => setRendered(rendered - 1)} >last Protein
          </ScrollButton> : <div></div> }
        <ColorButton
          onClick={() => setColorBy('aminos')} >Color by Aminos
        </ColorButton>
        <ColorButton
          onClick={() => setColorBy('order')} >Color by Order
        </ColorButton>
        <ColorButton
          onClick={() => setLegend(!legend)} >Turn on legend
        </ColorButton>
        {rendered - protiens.length ?
          <ScrollButton
            onClick={() => setRendered(rendered + 1)} >next Protein
          </ScrollButton> : <div></div>}
      </Footer>
    </RendererSection>


  )
}

const ScrollButton = styled.button`
  padding: 0.5em;
`
const ColorButton = styled.button`
  padding: 0.4em;
`
const Spacer = styled.div`
  width: 100%;
  height: 600px;
`
const Title = styled.h4`
  top: 1em;
  color: white;
  position: absolute;
`

const RendererSection = styled.section`
  width: 100%;
  height: 100%;
  display: flex;
  position: relative;
  align-items: center;
  flex-direction: column;
  justify-content: center;
`

const Footer = styled.section`
  width: 100%;
  height: 50px;
  bottom: 0px;
  display: flex;
  padding-top: 2em;
  align-items: center;
  padding-bottom: 2em;
  justify-content: space-around;
`
