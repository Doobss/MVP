import Plotly from 'plotly.js-gl3d-dist-min'


const marker = {
  size: 8,
  symbol: 'circle',
  line: {
    color: 'rgb(204, 204, 204)',
    width: 1
  },
  opacity: 0.8
}




const createPlot = (protien, colorBy = 'order', legend = false) => {

  const columns = splitIntoColumns(protien)
  marker.color = protien[colorBy]
  // console.log(protien['aminos'])
  var trace;
  if (legend) {
    const text = protien['aminos'].map(amino => `amino: ${amino}`)
    trace = {
      x: columns[0], y: columns[1], z: columns[2],
      // mode: 'markers',
      marker,
      type: 'scatter3d',
      mode: 'markers+text',
      name: 'Aminos',
      text,
      textfont : {
        family:'Times New Roman',
        size: 10,
      },
      textposition: 'bottom center',

    };
  }
  else {
    trace = {
      x: columns[0], y: columns[1], z: columns[2],
      marker,
      type: 'scatter3d',
      mode: 'markers',
      name: 'Aminos',
    };
  }

  var data = [trace];
  var layout = {
    height: 1050,
    margin: { l: 0, r: 0, b: 0, t: 0 },
    legend: {
      yref: 'paper',
      font: {
        family: 'Arial, sans-serif',
        size: 5,
        color: 'grey',
      }
    },
  };
    return Plotly.newPlot('renderer', data, layout);

}


export const splitIntoColumns = (protien) => {

  const x = []
  const y = []
  const z = []
  const protL = protien.vecs ? protien.vecs.length : 0
  let i = 0
  while (protL > i) {
    const row = protien.vecs[i]
    // console.log(row)
    x.push(row[0])
    y.push(row[1])
    z.push(row[2])
    i++
  }
  return [x, y, z]
}

export const renderProtien = (protein, colorBy, legend) => {
  return new Promise((res, rej) => res(createPlot(protein, colorBy, legend)))
}





const protSort = (a, b) => {
  return b.vecs.length - a.vecs.length
}

const sortProtiens = (protiens) => {
  return protiens.sort(protSort)
}

// 54
const bases = [4, 4, 4]

var breakDownCodon = (codon) => {
  let reduced = codon
  return bases.map((baseValue, ind) => {
    const denom = ((bases.length - ind) -1) * baseValue
    const newBase = Math.floor(reduced / denom)
    console.log({ reduced, denom, newBase })
    reduced -= denom * newBase
    return newBase
  })
}

var breakDownCodon = (codon) => {
  let reduced = codon
  return bases.reduce((memo, baseValue, ind) => {

    const mult = Math.floor(reduced / baseValue)
    const maxMult = ((bases.length - ind) -1) * baseValue
    let baseMult = mult > maxMult ? maxMult : mult
    const newBase = Math.floor(reduced / baseMult)
    console.log({ reduced, newBase, mult, maxMult, baseMult })
    memo.push(newBase)
    reduced -= baseMult * newBase
    return memo

  }, [])
}


const breakDownCodons = (codonChains) => {
  return codonChains.map(codons => {
    const filtered = codons.filter(value => value > -1)
    return filtered.map((codon) => {
      const newBases = breakDownCodon(codon)
      console.log(codon)
      console.log(newBases)
      return newBases
    })
  })
}


export const buildProtiens = (created) => {
  let [ createdProtiens, aminoChains, dna, codons ] = created
  const proteins =  createdProtiens.map((vecs, ind) => {
    const aminos = aminoChains[ind].filter(val => val > -1)
    return { vecs, aminos, order: [...Array(aminos.length).keys()] }
  })
  return sortProtiens(proteins)
}




