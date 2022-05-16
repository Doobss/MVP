
const buildGetOptions = (params = {}) => {
  return {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json'
    },
  }
}


const buildPostOptions = (params = {}, data = {}) => {
  return {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data) // body data type must match "Content-Type" header
  }
}


const BASE = 'http://localhost:5000'
const runFetch = (endpoint, options) => {
  return fetch(BASE + endpoint, options)
}


const get = (endpoint, params) => {
  return runFetch(endpoint, buildGetOptions(params))
          .then(res => res.json())
          .catch(err => console.log('get data err ', endpoint, err.message))
}

const post = (endpoint, params, data) => {
  return runFetch(endpoint, buildPostOptions(params, data))
}

const getCreate = (options) => {
  return get('/api/create', options)
}

get.create = getCreate
const api = {
  get,
  post
}

export default api


