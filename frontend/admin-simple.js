// Simplified Admin Panel JavaScript - CRUD, Procedures, Queries, Status
const API_BASE = 'http://localhost:8000';

// Tab management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
    
    // Load tab-specific data
    if (tabName === 'crud') {
        loadCategories();
    } else if (tabName === 'procedures') {
        loadCategories(); // For batch operations
    } else if (tabName === 'status') {
        refreshSystemStatus();
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Check authentication first
    const sessionId = localStorage.getItem('sessionId');
    const userRole = localStorage.getItem('userRole');
    const username = localStorage.getItem('username');

    if (!sessionId || userRole !== 'admin') {
        window.location.href = 'login.html';
        return;
    }

    // Update header with username
    const welcomeElement = document.getElementById('welcomeMessage');
    if (welcomeElement) {
        welcomeElement.textContent = `Welcome, Admin ${username}`;
    }
    
    loadCategories();
    refreshSystemStatus();
});

// Utility functions
function showMessage(message, type = 'success') {
    const className = type === 'error' ? 'error-message' : 'success-message';
    return `<div class="${className}">${message}</div>`;
}

function showLoading(containerId) {
    document.getElementById(containerId).innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';
}

async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(API_BASE + endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Load categories for dropdowns
async function loadCategories() {
    try {
        const categories = await apiCall('/categories');
        
        // Populate category dropdowns
        const selectors = ['updateImageCategory', 'batchCategoryId', 'imageCategory'];
        selectors.forEach(id => {
            const select = document.getElementById(id);
            if (select) {
                select.innerHTML = '<option value="">Select category...</option>';
                categories.categories.forEach(cat => {
                    select.innerHTML += `<option value="${cat.category_id}">${cat.category_name}</option>`;
                });
            }
        });
    } catch (error) {
        console.error('Error loading categories:', error);
    }
}

// === CRUD Operations ===

async function uploadImage() {
    const fileInput = document.getElementById('imageFile');
    const categoryId = document.getElementById('imageCategory').value;
    
    if (!fileInput.files[0]) {
        document.getElementById('crudResults').innerHTML = showMessage('Please select an image file', 'error');
        return;
    }
    
    if (!categoryId) {
        document.getElementById('crudResults').innerHTML = showMessage('Please select a category', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('category_id', categoryId);
    
    try {
        showLoading('crudResults');
        
        const response = await fetch(API_BASE + '/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const result = await response.json();
        
        fileInput.value = '';
        document.getElementById('imageCategory').value = '';
        document.getElementById('crudResults').innerHTML = showMessage(`Image uploaded successfully! ID: ${result.image_id}. Vector computed and indexed automatically.`);
    } catch (error) {
        document.getElementById('crudResults').innerHTML = showMessage(`Error: ${error.message}`, 'error');
    }
}

async function updateImage() {
    const imageId = document.getElementById('updateImageId').value;
    const categoryId = document.getElementById('updateImageCategory').value;
    
    if (!imageId) {
        document.getElementById('crudResults').innerHTML = showMessage('Please enter an image ID', 'error');
        return;
    }
    
    if (!categoryId) {
        document.getElementById('crudResults').innerHTML = showMessage('Please select a category', 'error');
        return;
    }
    
    try {
        await apiCall(`/admin/images/${imageId}`, {
            method: 'PUT',
            body: JSON.stringify({ category_id: parseInt(categoryId) })
        });
        
        document.getElementById('crudResults').innerHTML = showMessage('Image updated successfully!');
        document.getElementById('updateImageId').value = '';
    } catch (error) {
        document.getElementById('crudResults').innerHTML = showMessage(`Error: ${error.message}`, 'error');
    }
}

async function deleteImage() {
    const imageId = document.getElementById('deleteImageId').value;
    
    if (!imageId || !confirm('Are you sure you want to delete this image?')) {
        return;
    }
    
    try {
        await apiCall(`/admin/images/${imageId}`, {
            method: 'DELETE'
        });
        
        document.getElementById('crudResults').innerHTML = showMessage('Image deleted successfully!');
        document.getElementById('deleteImageId').value = '';
    } catch (error) {
        document.getElementById('crudResults').innerHTML = showMessage(`Error: ${error.message}`, 'error');
    }
}

async function deleteCategory() {
    const categoryId = document.getElementById('deleteCategoryId').value;
    
    if (!categoryId || !confirm('Are you sure you want to delete this category? This will also delete all images in this category.')) {
        return;
    }
    
    try {
        await apiCall(`/admin/categories/${categoryId}`, {
            method: 'DELETE'
        });
        
        document.getElementById('crudResults').innerHTML = showMessage('Category deleted successfully!');
        document.getElementById('deleteCategoryId').value = '';
        loadCategories();
    } catch (error) {
        document.getElementById('crudResults').innerHTML = showMessage(`Error: ${error.message}`, 'error');
    }
}

async function loadAllCategories() {
    try {
        showLoading('crudResults');
        const categories = await apiCall('/categories');
        
        let html = '<h3>All Categories</h3>';
        html += '<div class="table-container"><table><thead><tr>';
        html += '<th>ID</th><th>Name</th><th>Image Count</th>';
        html += '</tr></thead><tbody>';
        
        categories.categories.forEach(cat => {
            html += `<tr>
                <td>${cat.category_id}</td>
                <td>${cat.category_name}</td>
                <td>${cat.image_count}</td>
            </tr>`;
        });
        
        html += '</tbody></table></div>';
        document.getElementById('crudResults').innerHTML = html;
    } catch (error) {
        document.getElementById('crudResults').innerHTML = showMessage(`Error: ${error.message}`, 'error');
    }
}

async function loadRecentImages() {
    try {
        showLoading('crudResults');
        const images = await apiCall('/images?limit=20');
        
        let html = '<h3>Recent Images</h3>';
        html += '<div class="table-container"><table><thead><tr>';
        html += '<th>ID</th><th>Category</th><th>Actions</th>';
        html += '</tr></thead><tbody>';
        
        images.images.forEach(img => {
            html += `<tr>
                <td>${img.image_id}</td>
                <td>${img.category_name}</td>
                <td><button class="btn btn-danger" onclick="deleteImageById(${img.image_id})">Delete</button></td>
            </tr>`;
        });
        
        html += '</tbody></table></div>';
        document.getElementById('crudResults').innerHTML = html;
    } catch (error) {
        document.getElementById('crudResults').innerHTML = showMessage(`Error: ${error.message}`, 'error');
    }
}

async function deleteImageById(imageId) {
    if (!confirm('Are you sure you want to delete this image?')) return;
    
    try {
        await apiCall(`/admin/images/${imageId}`, {
            method: 'DELETE'
        });
        loadRecentImages(); // Reload the list
    } catch (error) {
        document.getElementById('crudResults').innerHTML = showMessage(`Error: ${error.message}`, 'error');
    }
}

async function loadSystemStats() {
    try {
        showLoading('crudResults');
        const status = await apiCall('/status');
        
        let html = '<h3>System Statistics</h3>';
        html += '<div class="row">';
        html += `<div class="col">
            <div class="card">
                <h4>Images</h4>
                <div style="font-size: 2rem; color: #667eea; font-weight: bold;">${status.total_images}</div>
            </div>
        </div>`;
        html += `<div class="col">
            <div class="card">
                <h4>Vectors</h4>
                <div style="font-size: 2rem; color: #667eea; font-weight: bold;">${status.total_vectors}</div>
            </div>
        </div>`;
        html += `<div class="col">
            <div class="card">
                <h4>Categories with Prototypes</h4>
                <div style="font-size: 2rem; color: #667eea; font-weight: bold;">${status.categories_with_prototypes}</div>
            </div>
        </div>`;
        html += '</div>';
        
        document.getElementById('crudResults').innerHTML = html;
    } catch (error) {
        document.getElementById('crudResults').innerHTML = showMessage(`Error: ${error.message}`, 'error');
    }
}

// === Stored Procedures ===

async function executeProcedure(procedureName) {
    try {
        showLoading('procedureResults');
        
        let params = {};
        
        // Skip problematic procedures
        if (procedureName === 'get_category_statistics') {
            document.getElementById('procedureResults').innerHTML = showMessage('Category statistics procedure is not available. Available procedures: Database Health, Cleanup Old Data, Rebuild Indexes.', 'error');
            return;
        }
        
        const result = await apiCall('/admin/procedures/execute', {
            method: 'POST',
            body: JSON.stringify({
                procedure_name: procedureName,
                parameters: params
            })
        });
        
        displayProcedureResult(result, procedureName);
    } catch (error) {
        document.getElementById('procedureResults').innerHTML = showMessage(`Error: ${error.message}`, 'error');
    }
}

async function executeAnalysisProcedure() {
    document.getElementById('procedureResults').innerHTML = showMessage('Category analysis is not available. Use Database Health check for system analysis.', 'info');
}

async function executeBatchOperation() {
    const categoryId = document.getElementById('batchCategoryId').value;
    
    if (!categoryId) {
        document.getElementById('procedureResults').innerHTML = showMessage('Please select a category', 'error');
        return;
    }
    
    try {
        showLoading('procedureResults');
        const result = await apiCall('/admin/batch-process', {
            method: 'POST',
            body: JSON.stringify({
                category_id: parseInt(categoryId),
                operation: 'validate',
                batch_size: 50
            })
        });
        
        let html = '<h3>Batch Operation Results</h3>';
        html += '<div class="table-container"><table><thead><tr>';
        html += '<th>Image ID</th><th>Result</th><th>Processing Time</th>';
        html += '</tr></thead><tbody>';
        
        result.results.forEach(item => {
            const statusClass = item.operation_result.includes('ERROR') ? 'status-error' : 'status-success';
            html += `<tr>
                <td>${item.image_id}</td>
                <td><span class="status-badge ${statusClass}">${item.operation_result}</span></td>
                <td>${item.processing_time}</td>
            </tr>`;
        });
        
        html += '</tbody></table></div>';
        document.getElementById('procedureResults').innerHTML = html;
    } catch (error) {
        document.getElementById('procedureResults').innerHTML = showMessage(`Error: ${error.message}`, 'error');
    }
}

function displayProcedureResult(result, procedureName) {
    let html = `<h3>✅ ${procedureName.replace('_', ' ').toUpperCase()}</h3>`;
    html += `<p><strong>⏱️ Execution Time:</strong> ${result.execution_time}ms</p>`;
    
    if (result.data && result.data.length > 0) {
        const columns = Object.keys(result.data[0]);
        
        html += '<div class="table-container"><table><thead><tr>';
        columns.forEach(col => {
            html += `<th>${col.replace('_', ' ').toUpperCase()}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        result.data.forEach(row => {
            html += '<tr>';
            columns.forEach(col => {
                const value = row[col];
                let displayValue = value;
                
                if (typeof value === 'object') {
                    displayValue = JSON.stringify(value, null, 2);
                } else if (typeof value === 'boolean') {
                    displayValue = `<span class="status-badge ${value ? 'status-success' : 'status-error'}">${value ? '✅ YES' : '❌ NO'}</span>`;
                } else if (col === 'metric_name' && value) {
                    displayValue = `📊 ${value}`;
                } else if (col === 'status' && value) {
                    const statusClass = value === 'OK' ? 'status-success' : value === 'WARNING' ? 'status-info' : 'status-error';
                    displayValue = `<span class="status-badge ${statusClass}">${value}</span>`;
                }
                
                html += `<td>${displayValue}</td>`;
            });
            html += '</tr>';
        });
        
        html += '</tbody></table></div>';
    } else {
        html += showMessage('✅ Procedure executed successfully - no data returned');
    }
    
    document.getElementById('procedureResults').innerHTML = html;
}



// === System Status ===

async function refreshSystemStatus() {
    try {
        showLoading('systemStatusResults');
        const status = await apiCall('/status');
        
        let html = '<div class="row">';
        html += `<div class="col">
            <div class="card">
                <h3>Database Status</h3>
                <p><strong>Total Images:</strong> ${status.total_images}</p>
                <p><strong>Total Vectors:</strong> ${status.total_vectors}</p>
                <p><strong>Categories with Prototypes:</strong> ${status.categories_with_prototypes}</p>
                <p><strong>FAISS Index:</strong> <span class="status-badge ${status.faiss_index_loaded ? 'status-success' : 'status-error'}">${status.faiss_index_loaded ? 'LOADED' : 'NOT LOADED'}</span></p>
            </div>
        </div>`;
        
        html += `<div class="col">
            <div class="card">
                <h3>System Health</h3>`;
        
        try {
            const health = await apiCall('/admin/health');
            health.health_metrics.forEach(metric => {
                const statusClass = metric.status === 'OK' ? 'status-success' : 
                                   metric.status === 'WARNING' ? 'status-info' : 'status-error';
                
                html += `<p><strong>${metric.metric_name}:</strong> ${metric.metric_value} <span class="status-badge ${statusClass}">${metric.status}</span></p>`;
            });
        } catch (e) {
            html += '<p>Health metrics unavailable</p>';
        }
        
        html += '</div></div></div>';
        
        document.getElementById('systemStatusResults').innerHTML = html;
    } catch (error) {
        document.getElementById('systemStatusResults').innerHTML = showMessage(`Error: ${error.message}`, 'error');
    }
}

// Logout function
function logout() {
    localStorage.clear();
    window.location.href = 'login.html';
}