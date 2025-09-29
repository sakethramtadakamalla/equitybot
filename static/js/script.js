document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('report-form');
    const sectorSelect = document.getElementById('sector-select');
    const stockSelect = document.getElementById('stock-select');
    const generateBtn = document.getElementById('generate-btn');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('loading-spinner');
    const resultContainer = document.getElementById('result-container');
    const errorContainer = document.getElementById('error-container');
    const statusContainer = document.getElementById('status-container');
    const downloadLink = document.getElementById('download-link');
    const errorMessage = document.getElementById('error-message');
    const statusText = document.getElementById('status-text');

    let statusInterval;
    const statusMessages = [
        "Initializing analysis...", "Fetching fundamental data...", "Calculating technical indicators...",
        "Scraping latest news...", "Compiling the report..."
    ];

    function startStatusUpdates() {
        let i = 0;
        statusContainer.classList.remove('d-none');
        statusText.textContent = statusMessages[0];
        statusInterval = setInterval(() => {
            i = (i + 1) % statusMessages.length;
            statusText.textContent = statusMessages[i];
        }, 2000);
    }

    function stopStatusUpdates() {
        clearInterval(statusInterval);
        statusContainer.classList.add('d-none');
    }

    async function populateSelectors() {
        try {
            const response = await fetch('/api/stocks');
            if (!response.ok) throw new Error('Failed to load stock data.');
            const stockData = await response.json();
            
            for (const sector in stockData) {
                const option = new Option(sector, sector);
                sectorSelect.add(option);
            }
            
            sectorSelect.addEventListener('change', () => {
                const selectedSector = sectorSelect.value;
                stockSelect.innerHTML = '<option selected disabled value="">2. Select a Stock</option>';
                stockSelect.disabled = true;
                generateBtn.disabled = true;
                if (selectedSector && stockData[selectedSector]) {
                    stockData[selectedSector].forEach(stock => {
                        const option = new Option(`${stock.name} (${stock.ticker})`, stock.ticker);
                        stockSelect.add(option);
                    });
                    stockSelect.disabled = false;
                }
            });

        } catch (error) {
            errorMessage.textContent = error.message;
            errorContainer.classList.remove('d-none');
        }
    }

    stockSelect.addEventListener('change', () => {
        generateBtn.disabled = !stockSelect.value;
    });

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        generateBtn.disabled = true;
        spinner.classList.remove('d-none');
        btnText.textContent = 'Processing...';
        resultContainer.classList.add('d-none');
        errorContainer.classList.add('d-none');
        startStatusUpdates();

        fetch('/generate', { method: 'POST', body: new FormData(form) })
            .then(response => response.json().then(data => ({ ok: response.ok, data })))
            .then(({ ok, data }) => {
                if (!ok) throw new Error(data.error);
                downloadLink.href = data.download_url;
                resultContainer.classList.remove('d-none');
            })
            .catch(error => {
                errorMessage.textContent = error.message || 'An unknown error occurred.';
                errorContainer.classList.remove('d-none');
            })
            .finally(() => {
                stopStatusUpdates();
                generateBtn.disabled = false;
                spinner.classList.add('d-none');
                btnText.textContent = 'Generate Report';
            });
    });

    populateSelectors();
});