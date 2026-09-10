//! Python 辅助函数
//! 
//! 提供一些方便的 Python 绑定辅助函数

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

/// 将 Vec<Tile> 转换为 Python 列表
pub fn tile_vec_to_py_list(py: Python, tiles: &[crate::tile::Tile]) -> PyResult<Py<PyList>> {
    let py_tiles: Vec<Py<PyAny>> = tiles.iter()
        .map(|tile| {
            let py_tile = PyTile { inner: *tile };
            Py::new(py, py_tile)?.into()
        })
        .collect();
    
    Ok(PyList::new(py, py_tiles).into())
}

/// 将 Python 列表转换为 Vec<Tile>
pub fn py_list_to_tile_vec(py_list: &PyList) -> PyResult<Vec<crate::tile::Tile>> {
    let mut tiles = Vec::new();
    
    for item in py_list.iter() {
        let py_tile: &PyTile = item.extract()?;
        tiles.push(py_tile.inner);
    }
    
    Ok(tiles)
}

/// 将 Vec<i32> 转换为 Python 列表
pub fn int_vec_to_py_list(py: Python, ints: &[i32]) -> PyResult<Py<PyList>> {
    Ok(PyList::new(py, ints).into())
}

/// 将 Vec<f32> 转换为 Python 列表
pub fn float_vec_to_py_list(py: Python, floats: &[f32]) -> PyResult<Py<PyList>> {
    Ok(PyList::new(py, floats).into())
}

/// 将 Python 元组转换为 Vec<f32>
pub fn py_tuple_to_float_vec(py_tuple: &PyTuple) -> PyResult<Vec<f32>> {
    let mut floats = Vec::new();
    
    for item in py_tuple.iter() {
        let float: f32 = item.extract()?;
        floats.push(float);
    }
    
    Ok(floats)
}

/// 将 Vec<f32> 转换为 Python 元组
pub fn float_vec_to_py_tuple(py: Python, floats: &[f32]) -> PyResult<Py<PyTuple>> {
    let py_floats: Vec<Py<PyAny>> = floats.iter()
        .map(|&f| PyFloat::new(py, f).into())
        .collect();
    
    Ok(PyTuple::new(py, py_floats).into())
}

/// 将 GameResult 转换为 Python 字典
pub fn game_result_to_py_dict(py: Python, result: &crate::arena::GameResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    
    dict.set_item("final_scores", int_vec_to_py_list(py, &result.final_scores)?)?;
    dict.set_item("winners", int_vec_to_py_list(py, &result.winners)?)?;
    dict.set_item("losers", int_vec_to_py_list(py, &result.losers)?)?;
    dict.set_item("total_turns", result.total_turns)?;
    dict.set_item("game_over_reason", result.game_over_reason.clone())?;
    
    Ok(dict.into())
}

/// 将 GameSummary 转换为 Python 字典
pub fn game_summary_to_py_dict(py: Python, summary: &crate::arena::GameSummary) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    
    dict.set_item("phase", format!("{:?}", summary.phase))?;
    dict.set_item("turn", summary.turn)?;
    dict.set_item("current_player", summary.current_player)?;
    dict.set_item("dealer_position", summary.dealer_position)?;
    dict.set_item("wall_remaining", summary.wall_remaining)?;
    dict.set_item("players_won", summary.players_won)?;
    dict.set_item("game_over", summary.game_over)?;
    dict.set_item("game_history_count", summary.game_history_count)?;
    
    Ok(dict.into())
}

/// 将 PlayerSummary 转换为 Python 字典
pub fn player_summary_to_py_dict(py: Python, summary: &crate::state::PlayerSummary) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    
    dict.set_item("id", summary.id)?;
    dict.set_item("hand_size", summary.hand_size)?;
    dict.set_item("meld_count", summary.meld_count)?;
    dict.set_item("discarded_count", summary.discarded_count)?;
    dict.set_item("has_won", summary.has_won)?;
    dict.set_item("has_huazhu", summary.has_huazhu)?;
    dict.set_item("has_dajiao", summary.has_dajiao)?;
    dict.set_item("score", summary.score)?;
    dict.set_item("fan", summary.fan)?;
    
    Ok(dict.into())
}

/// 错误处理辅助函数
pub fn handle_py_result<T>(result: Result<T, Box<dyn std::error::Error + Send + Sync>>) -> PyResult<T> {
    match result {
        Ok(value) => Ok(value),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// 性能监控装饰器
#[pyfunction]
pub fn monitor_performance(py: Python, func: PyObject) -> PyResult<PyObject> {
    let start = std::time::Instant::now();
    
    let result = func.call0(py)?;
    
    let duration = start.elapsed();
    println!("Function executed in {:?}", duration);
    
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tile_vec_conversion() {
        let tiles = vec![crate::tile::Tile::M1, crate::tile::Tile::M2];
        let py = Python::with_gil(|py| {
            let py_list = tile_vec_to_py_list(py, &tiles).unwrap();
            let converted_tiles = py_list_to_tile_vec(&py_list.as_ref(py)).unwrap();
            assert_eq!(converted_tiles, tiles);
        });
    }
    
    #[test]
    fn test_int_vec_conversion() {
        let ints = vec![1, 2, 3, 4, 5];
        let py = Python::with_gil(|py| {
            let py_list = int_vec_to_py_list(py, &ints).unwrap();
            let converted_ints: Vec<i32> = py_list.extract().unwrap();
            assert_eq!(converted_ints, ints);
        });
    }
    
    #[test]
    fn test_float_vec_conversion() {
        let floats = vec![1.0, 2.5, 3.7, 4.2];
        let py = Python::with_gil(|py| {
            let py_list = float_vec_to_py_list(py, &floats).unwrap();
            let converted_floats: Vec<f32> = py_list.extract().unwrap();
            assert_eq!(converted_floats, floats);
        });
    }
}