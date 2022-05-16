import React from 'react';
import { Scene } from 'three';
import styled from 'styled-components';
// import Plotly from 'plotly.js-gl3d-dist-min'
import { renderProtien } from '../methods'

const scene = new Scene();

export default function Renderer({ protiens }) {

  // console.log(splitIntoColumns(protiens[0]))
  renderProtien(protiens[0])
  const root = document.getElementById('root');
  const mediaWidth = root.clientWidth || 600;
  const mediaHeight = root.clientHeight || 600;

  return (
    <RendererSection >
      <RendererContainer  >
        <canvas height={500} width={mediaWidth * 0.90} id="scene"></canvas>
      </RendererContainer>
    </RendererSection>
  )
}

const RendererSection = styled.section`
  width: 100%;
  height: 100%;
  display: flex;
  padding-top: 2em;
  align-items: center;
  padding-bottom: 2em;
  justify-content: center;
`

const RendererContainer = styled.div`
  width: 90%;
  height: 90%;
  background-color: rgb(40, 40, 40);
`