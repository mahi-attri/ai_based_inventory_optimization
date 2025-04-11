// dashboard.js - JavaScript for Retail Inventory Optimizer Dashboard

// Initialize dashboard when the document is ready
document.addEventListener('DOMContentLoaded', function() {
    // References to dashboard elements
    const refreshBtn = document.getElementById('refreshBtn');
    const runAgentsBtn = document.getElementById('runAgentsBtn');
    const saveConfigBtn = document.getElementById('saveConfigBtn');
    const inventoryHealthChart = document.getElementById('inventoryHealthChart');
    const chartTabs = document.querySelectorAll('.tab');
    const runAgentsModal = document.getElementById('runAgentsModal');
    const cancelRunBtn = document.getElementById('cancelRunBtn');
    const confirmRunBtn = document.getElementById('confirmRunBtn');
    const modalClose = document.querySelector('.modal-close');
    const notificationCenter = document.querySelector('.notification-center');
    const accordionHeaders = document.querySelectorAll('.accordion-header');
    
    // Chart variables
    let inventoryHealthChartInstance = null;
    
    // Initialize dashboard on load
    initializeDashboard();
    
    // Dashboard initialization function
    function initializeDashboard() {
        // Load all data asynchronously
        Promise.all([
            fetchDashboardSummary(),
            fetchAgentStatus(),
            fetchInventoryAlerts(),
            fetchInventoryHealthData('week'),
            fetchRecentActivity(),
            fetchSystemConfig()
        ]).then(([summary, agentStatus, alerts, healthData, activity, config]) => {
            updateDashboardSummary(summary);
            updateAgentStatus(agentStatus);
            updateInventoryAlerts(alerts);
            createInventoryHealthChart(healthData);
            updateRecentActivity(activity);
            updateSystemConfig(config);
            
            // Remove any loading indicators
            document.body.classList.remove('loading');
        }).catch(error => {
            console.error('Error initializing dashboard:', error);
            showNotification('Error', 'Failed to load dashboard data. Please try again.', 'danger');
        });
        
        // Set up event listeners
        setupEventListeners();
    }
    
    // Function to fetch dashboard summary data
    function fetchDashboardSummary() {
        return fetch('/api/dashboard/summary')
            .then(response => response.json())
            .catch(error => {
                console.error('Error fetching dashboard summary:', error);
                return {
                    "out_of_stock": { "value": 24, "change": -12 },
                    "overstocked": { "value": 43, "change": -8 },
                    "weekly_lost_sales": { "value": 12350, "change": 5 }
                };
            });
    }
    
    // Function to fetch agent status
    function fetchAgentStatus() {
        return fetch('/api/agents/status')
            .then(response => response.json())
            .catch(error => {
                console.error('Error fetching agent status:', error);
                return {};
            });
    }
    
    // Function to fetch inventory alerts
    function fetchInventoryAlerts(limit = 4) {
        return fetch(`/api/inventory/alerts?limit=${limit}`)
            .then(response => response.json())
            .catch(error => {
                console.error('Error fetching inventory alerts:', error);
                return [];
            });
    }
    
    // Function to fetch inventory health data
    function fetchInventoryHealthData(period = 'week') {
        return fetch(`/api/inventory/health?period=${period}`)
            .then(response => response.json())
            .catch(error => {
                console.error('Error fetching inventory health data:', error);
                return {
                    labels: [],
                    datasets: []
                };
            });
    }
    
    // Function to fetch recent activity
    function fetchRecentActivity(limit = 5) {
        return fetch(`/api/activity/recent?limit=${limit}`)
            .then(response => response.json())
            .catch(error => {
                console.error('Error fetching recent activity:', error);
                return [];
            });
    }
    
    // Function to fetch system configuration
    function fetchSystemConfig() {
        return fetch('/api/config')
            .then(response => response.json())
            .catch(error => {
                console.error('Error fetching system config:', error);
                return {
                    system: {
                        run_mode: 'continuous',
                        run_interval_seconds: 300,
                        ollama_base_url: 'http://localhost:11434',
                        log_level: 'INFO'
                    }
                };
            });
    }
    
    // Function to update dashboard summary
    function updateDashboardSummary(summary) {
        // Update out of stock card
        const outOfStockValueEl = document.querySelector('.card:nth-child(1) .card-value');
        const outOfStockTrendEl = document.querySelector('.card:nth-child(1) .card-trend');
        
        if (outOfStockValueEl) {
            outOfStockValueEl.textContent = summary.out_of_stock.value;
        }
        
        if (outOfStockTrendEl) {
            const trend = summary.out_of_stock.change;
            outOfStockTrendEl.innerHTML = `
                <i class="icon">${trend < 0 ? 'arrow_downward' : 'arrow_upward'}</i>
                <span>${Math.abs(trend)}% from last week</span>
            `;
            outOfStockTrendEl.className = `card-trend ${trend < 0 ? 'trend-down' : 'trend-up'}`;
        }
        
        // Update overstocked card
        const overstockedValueEl = document.querySelector('.card:nth-child(2) .card-value');
        const overstockedTrendEl = document.querySelector('.card:nth-child(2) .card-trend');
        
        if (overstockedValueEl) {
            overstockedValueEl.textContent = summary.overstocked.value;
        }
        
        if (overstockedTrendEl) {
            const trend = summary.overstocked.change;
            overstockedTrendEl.innerHTML = `
                <i class="icon">${trend < 0 ? 'arrow_downward' : 'arrow_upward'}</i>
                <span>${Math.abs(trend)}% from last week</span>
            `;
            overstockedTrendEl.className = `card-trend ${trend < 0 ? 'trend-down' : 'trend-up'}`;
        }
        
        // Update weekly lost sales card
        const lostSalesValueEl = document.querySelector('.card:nth-child(3) .card-value');
        const lostSalesTrendEl = document.querySelector('.card:nth-child(3) .card-trend');
        
        if (lostSalesValueEl) {
            lostSalesValueEl.textContent = `$${summary.weekly_lost_sales.value.toLocaleString()}`;
        }
        
        if (lostSalesTrendEl) {
            const trend = summary.weekly_lost_sales.change;
            lostSalesTrendEl.innerHTML = `
                <i class="icon">${trend < 0 ? 'arrow_downward' : 'arrow_upward'}</i>
                <span>${Math.abs(trend)}% from last week</span>
            `;
            lostSalesTrendEl.className = `card-trend ${trend < 0 ? 'trend-down' : 'trend-up'}`;
        }
    }
    
    // Function to update agent status
    function updateAgentStatus(agentStatus) {
        const agentGrid = document.querySelector('.agent-grid');
        
        if (!agentGrid) return;
        
        // Clear existing agent cards
        agentGrid.innerHTML = '';
        
        // Create a card for each agent
        for (const [agentId, agent] of Object.entries(agentStatus)) {
            // Determine status class
            let statusClass = 'status-offline';
            if (agent.status === 'active') {
                statusClass = 'status-active';
            } else if (agent.status === 'idle') {
                statusClass = 'status-idle';
            }
            
            // Create metrics HTML
            let metricsHtml = '';
            for (const [key, value] of Object.entries(agent.metrics)) {
                const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                metricsHtml += `
                    <div class="agent-stats">
                        <div class="agent-stat-label">${formattedKey}:</div>
                        <div class="agent-stat-value">${value}</div>
                    </div>
                `;
            }
            
            // Create button based on agent status
            const buttonHtml = agent.status === 'idle' ? 
                '<button class="btn btn-outline" onclick="runSingleAgent(\'' + agentId + '\')">Run Now</button>' : 
                '<button class="btn btn-outline">Details</button>';
            
            // Create agent card
            const agentCard = document.createElement('div');
            agentCard.className = 'agent-card';
            agentCard.innerHTML = `
                <div class="agent-header">
                    <div class="agent-title">${agent.name}</div>
                    <div class="agent-status">
                        <div class="status-dot ${statusClass}"></div>
                        <span>${agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}</span>
                    </div>
                </div>
                <div class="agent-body">
                    <div class="agent-stats">
                        <div class="agent-stat-label">Last Run:</div>
                        <div class="agent-stat-value">${agent.last_run}</div>
                    </div>
                    ${metricsHtml}
                </div>
                <div class="agent-footer">
                    ${buttonHtml}
                </div>
            `;
            
            agentGrid.appendChild(agentCard);
        }
    }
    
    // Function to update inventory alerts
    function updateInventoryAlerts(alerts) {
        const tableBody = document.querySelector('.table-container .table tbody');
        
        if (!tableBody) return;
        
        // Clear existing alerts
        tableBody.innerHTML = '';
        
        if (alerts.length === 0) {
            const tr = document.createElement('tr');
            tr.innerHTML = '<td colspan="7" style="text-align: center;">No inventory alerts found.</td>';
            tableBody.appendChild(tr);
            return;
        }
        
        // Add each alert to the table
        alerts.forEach(alert => {
            const tr = document.createElement('tr');
            
            // Determine badge class based on status
            let badgeClass = 'badge-success';
            if (alert.status === 'Out of Stock') {
                badgeClass = 'badge-danger';
            } else if (alert.status === 'Low Stock' || alert.status === 'Overstocked') {
                badgeClass = 'badge-warning';
            }
            
            // Determine action button based on status
            let actionButton = '<button class="btn btn-outline">Details</button>';
            if (alert.status === 'Out of Stock' || alert.status === 'Low Stock') {
                actionButton = '<button class="btn btn-outline">Order</button>';
            } else if (alert.status === 'Overstocked') {
                actionButton = '<button class="btn btn-outline">Discount</button>';
            }
            
            tr.innerHTML = `
                <td>${alert.product}</td>
                <td>${alert.category}</td>
                <td>${alert.store}</td>
                <td><span class="badge ${badgeClass}">${alert.status}</span></td>
                <td>${alert.current_stock}</td>
                <td>${alert.recommended}</td>
                <td>${actionButton}</td>
            `;
            
            tableBody.appendChild(tr);
        });
    }
    
    // Function to create inventory health chart
    function createInventoryHealthChart(data) {
        // Destroy existing chart if it exists
        if (inventoryHealthChartInstance) {
            inventoryHealthChartInstance.destroy();
        }
        
        // Create new chart
        inventoryHealthChartInstance = new Chart(
            inventoryHealthChart,
            {
                type: 'bar',
                data: data,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                    },
                    scales: {
                        x: {
                            stacked: true,
                        },
                        y: {
                            stacked: true,
                            title: {
                                display: true,
                                text: 'Number of Products'
                            }
                        }
                    }
                }
            }
        );
    }
    
    // Function to update recent activity
    function updateRecentActivity(activity) {
        const tableBody = document.querySelector('.accordion-body.show .table tbody');
        
        if (!tableBody) return;
        
        // Clear existing activity
        tableBody.innerHTML = '';
        
        if (activity.length === 0) {
            const tr = document.createElement('tr');
            tr.innerHTML = '<td colspan="4" style="text-align: center;">No recent activity found.</td>';
            tableBody.appendChild(tr);
            return;
        }
        
        // Add each activity to the table
        activity.forEach(log => {
            const tr = document.createElement('tr');
            
            // Determine badge class based on status
            let badgeClass = 'badge-success';
            if (log.status === 'error' || log.status === 'failed') {
                badgeClass = 'badge-danger';
            } else if (log.status === 'warning' || log.status === 'alert') {
                badgeClass = 'badge-warning';
            }
            
            tr.innerHTML = `
                <td>${log.time}</td>
                <td>${log.agent}</td>
                <td>${log.activity}</td>
                <td><span class="badge ${badgeClass}">${log.status.charAt(0).toUpperCase() + log.status.slice(1)}</span></td>
            `;
            
            tableBody.appendChild(tr);
        });
    }
    
    // Function to update system configuration
    function updateSystemConfig(config) {
        const runModeSelect = document.getElementById('runModeSelect');
        const runIntervalInput = document.getElementById('runIntervalInput');
        const ollamaUrlInput = document.getElementById('ollamaUrlInput');
        const logLevelSelect = document.getElementById('logLevelSelect');
        
        if (config.system) {
            if (runModeSelect && config.system.run_mode) {
                runModeSelect.value = config.system.run_mode;
            }
            
            if (runIntervalInput && config.system.run_interval_seconds) {
                runIntervalInput.value = config.system.run_interval_seconds;
            }
            
            if (ollamaUrlInput && config.system.ollama_base_url) {
                ollamaUrlInput.value = config.system.ollama_base_url;
            }
            
            if (logLevelSelect && config.system.log_level) {
                logLevelSelect.value = config.system.log_level;
            }
        }
    }
    
    // Function to set up event listeners
    function setupEventListeners() {
        // Refresh button
        if (refreshBtn) {
            refreshBtn.addEventListener('click', function() {
                // Show a loading indicator
                this.innerHTML = '<i class="icon">sync</i><span>Refreshing...</span>';
                this.disabled = true;
                
                // Refresh dashboard data
                initializeDashboard();
                
                // Reset the button after delay
                setTimeout(() => {
                    this.innerHTML = '<i class="icon">refresh</i><span>Refresh</span>';
                    this.disabled = false;
                    
                    // Show a notification
                    showNotification('Dashboard Refreshed', 'The dashboard has been updated with the latest data.', 'success');
                }, 1500);
            });
        }
        
        // Save configuration button
        if (saveConfigBtn) {
            saveConfigBtn.addEventListener('click', function() {
                // Get configuration values
                const runMode = document.getElementById('runModeSelect').value;
                const runInterval = parseInt(document.getElementById('runIntervalInput').value);
                const ollamaUrl = document.getElementById('ollamaUrlInput').value;
                const logLevel = document.getElementById('logLevelSelect').value;
                
                // Validate inputs
                if (runInterval < 1) {
                    showNotification('Error', 'Run interval must be at least 1 second.', 'danger');
                    return;
                }
                
                if (!ollamaUrl) {
                    showNotification('Error', 'Ollama Base URL is required.', 'danger');
                    return;
                }
                
                // Prepare new config
                const newConfig = {
                    system: {
                        run_mode: runMode,
                        run_interval_seconds: runInterval,
                        ollama_base_url: ollamaUrl,
                        log_level: logLevel
                    }
                };
                
                // Show loading state
                this.innerHTML = '<i class="icon">sync</i><span>Saving...</span>';
                this.disabled = true;
                
                // Send to API
                fetch('/api/config', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(newConfig)
                })
                .then(response => response.json())
                .then(data => {
                    // Reset button
                    this.innerHTML = '<i class="icon">save</i><span>Save Configuration</span>';
                    this.disabled = false;
                    
                    if (data.status === 'success') {
                        showNotification('Configuration Saved', 'System configuration has been updated.', 'success');
                    } else {
                        showNotification('Error', `Failed to save configuration: ${data.message}`, 'danger');
                    }
                })
                .catch(error => {
                    // Reset button
                    this.innerHTML = '<i class="icon">save</i><span>Save Configuration</span>';
                    this.disabled = false;
                    
                    console.error('Error saving configuration:', error);
                    showNotification('Error', 'Failed to save configuration. Please try again.', 'danger');
                });
            });
        }
        
        // Chart tabs
        chartTabs.forEach(tab => {
            tab.addEventListener('click', function() {
                // Remove active class from all tabs
                chartTabs.forEach(t => t.classList.remove('active'));
                // Add active class to clicked tab
                this.classList.add('active');
                
                // Get the period from the tab
                const period = this.getAttribute('data-period');
                
                // Fetch new chart data for the selected period
                fetchInventoryHealthData(period)
                    .then(data => {
                        createInventoryHealthChart(data);
                    })
                    .catch(error => {
                        console.error(`Error fetching ${period} inventory health data:`, error);
                    });
            });
        });
        
        // Run agents button
        if (runAgentsBtn) {
            runAgentsBtn.addEventListener('click', function() {
                runAgentsModal.style.display = 'flex';
            });
        }
        
        // Cancel run button
        if (cancelRunBtn) {
            cancelRunBtn.addEventListener('click', closeModal);
        }
        
        // Modal close button
        if (modalClose) {
            modalClose.addEventListener('click', closeModal);
        }
        
        // Confirm run button
        if (confirmRunBtn) {
            confirmRunBtn.addEventListener('click', function() {
                // Get selected agents
                const checkboxes = document.querySelectorAll('#runAgentsModal input[type="checkbox"]');
                const selectedAgents = [];
                let runAll = false;
                
                checkboxes.forEach(checkbox => {
                    if (checkbox.checked) {
                        const agentName = checkbox.parentElement.textContent.trim();
                        if (agentName === 'All Agents') {
                            runAll = true;
                        } else {
                            // Convert display name to agent ID format
                            const agentId = agentName.toLowerCase().replace(/\s+/g, '_') + (agentName !== 'Coordinator' ? '_agent' : '');
                            selectedAgents.push(agentId);
                        }
                    }
                });
                
                // Get run mode
                const runMode = document.querySelector('#runAgentsModal .form-select').value;
                
                // Close modal
                closeModal();
                
                // Show loading notification
                showNotification('Running Agents', 'Starting the selected agents...', 'warning');
                
                // Call API to run agents
                fetch('/api/agents/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        agents: runAll ? ['all'] : selectedAgents,
                        mode: runMode
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        showNotification('Agents Running', 'The selected agents have been started successfully.', 'success');
                        
                        // Refresh dashboard after a delay
                        setTimeout(() => {
                            initializeDashboard();
                        }, 2000);
                    } else {
                        showNotification('Error', `Failed to run agents: ${data.message}`, 'danger');
                    }
                })
                .catch(error => {
                    console.error('Error running agents:', error);
                    showNotification('Error', 'Failed to run agents. Please try again.', 'danger');
                });
            });
        }
        
        // Accordion headers
        accordionHeaders.forEach(header => {
            header.addEventListener('click', function() {
                const body = this.nextElementSibling;
                body.classList.toggle('show');
                
                // Toggle the icon
                const icon = this.querySelector('.icon');
                if (body.classList.contains('show')) {
                    icon.textContent = 'expand_less';
                } else {
                    icon.textContent = 'expand_more';
                }
            });
        });
        
        // Notification close buttons
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('notification-close')) {
                e.target.closest('.notification').remove();
            }
        });
    }
    
    // Function to close the modal
    function closeModal() {
        runAgentsModal.style.display = 'none';
    }
    
    // Function to show a notification
    window.showNotification = function(title, message, type) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        
        notification.innerHTML = `
            <div class="notification-icon">
                <i class="icon">${type === 'success' ? 'check_circle' : type === 'warning' ? 'warning' : 'error'}</i>
            </div>
            <div class="notification-content">
                <div class="notification-title">${title}</div>
                <div class="notification-message">${message}</div>
            </div>
            <button class="notification-close">&times;</button>
        `;
        
        notificationCenter.appendChild(notification);
        
        // Automatically remove the notification after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    };
    
    // Function to run a single agent
    window.runSingleAgent = function(agentId) {
        // Show loading notification
        showNotification('Running Agent', `Starting ${agentId.replace('_agent', '').replace('_', ' ')}...`, 'warning');
        
        // Call API to run the agent
        fetch('/api/agents/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                agents: [agentId],
                mode: 'single'
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showNotification('Agent Running', 'The agent has been started successfully.', 'success');
                
                // Refresh dashboard after a delay
                setTimeout(() => {
                    initializeDashboard();
                }, 2000);
            } else {
                showNotification('Error', `Failed to run agent: ${data.message}`, 'danger');
            }
        })
        .catch(error => {
            console.error('Error running agent:', error);
            showNotification('Error', 'Failed to run agent. Please try again.', 'danger');
        });
    };
});