// المتغيرات العامة
let currentDataset = null;
let currentTarget = null;
let methodResults = {};
let gaResult = null;

$(document).ready(function() {
    console.log("✅ الصفحة محملة بنجاح");
    checkServerHealth();
    checkCacheStatus();
    setupEventHandlers();
});

function checkServerHealth() {
    $.get('/api/health')
        .done(function(response) {
            console.log("✅ السيرفر يعمل بشكل صحيح");
        })
        .fail(function() {
            showError('❌ لا يمكن الاتصال بالسيرفر. تأكد من تشغيل السيرفر.');
        });
}

function setupEventHandlers() {
    $('#generateData').click(generateData);
    $('#uploadData').click(uploadData);
    $('#fetchData').click(fetchData);
    $('#runGA').click(runGeneticAlgorithm);
    $('#clearCache').click(clearAllCache);
    $('#compareMethods').click(compareMethods);
    $('#runTraditionalMethods').click(runAllTraditionalMethods);
    $('#runStatisticalMethods').click(runAllStatisticalMethods);
}

function checkCacheStatus() {
    $.get('/api/cache/status')
        .done(function(response) {
            if (response.has_ga) {
                $('#gaCacheStatus').text('مخزنة').removeClass('bg-secondary').addClass('bg-success');
                $('#gaCacheStatus2').text('مخزنة').removeClass('bg-secondary').addClass('bg-success');
            } else {
                $('#gaCacheStatus').text('غير متوفرة').removeClass('bg-success').addClass('bg-secondary');
                $('#gaCacheStatus2').text('غير متوفرة').removeClass('bg-success').addClass('bg-secondary');
            }
            
            if (response.traditional_methods_count > 0) {
                $('#traditionalCacheStatus').text(`${response.traditional_methods_count} طريقة مخزنة`)
                    .removeClass('bg-secondary').addClass('bg-info');
            } else {
                $('#traditionalCacheStatus').text('لا توجد نتائج مخزنة')
                    .removeClass('bg-info').addClass('bg-secondary');
            }
        })
        .fail(function() {
            $('#gaCacheStatus, #gaCacheStatus2').text('خطأ في الاتصال').addClass('bg-danger');
            $('#traditionalCacheStatus').text('خطأ في الاتصال').addClass('bg-danger');
        });
}

function generateData() {
    const nSamples = $('#nSamples').val();
    const nFeatures = $('#nFeatures').val();
    const nInformative = $('#nInformative').val();
    
    console.log(`🔄 محاولة توليد بيانات: ${nSamples} عينة, ${nFeatures} ميزة, ${nInformative} معلوماتية`);
    
    showLoading($('#generateData'));
    
    $.ajax({
        url: '/api/generate',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            nSamples: parseInt(nSamples),
            nFeatures: parseInt(nFeatures),
            nInformative: parseInt(nInformative)
        }),
        success: function(response) {
            hideLoading($('#generateData'));
            if (response.error) {
                showError('خطأ في توليد البيانات: ' + response.error);
                return;
            }
            
            currentDataset = response.csv;
            currentTarget = response.target;
            displayDataPreview(response.csv);
            showSuccess(response.message || 'تم توليد البيانات بنجاح!');
            console.log("✅ تم توليد البيانات بنجاح");
        },
        error: function(xhr, status, error) {
            hideLoading($('#generateData'));
            let errorMsg = 'فشل في الاتصال بالخادم';
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMsg = xhr.responseJSON.error;
            }
            showError('خطأ في توليد البيانات: ' + errorMsg);
            console.error("❌ خطأ في توليد البيانات:", error);
        }
    });
}

function uploadData() {
    const fileInput = $('#fileUpload')[0];
    if (!fileInput.files.length) {
        showError('يرجى اختيار ملف أولاً');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    showLoading($('#uploadData'));
    
    $.ajax({
        url: '/api/upload',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            hideLoading($('#uploadData'));
            if (response.error) {
                showError('خطأ في رفع الملف: ' + response.error);
                return;
            }
            
            currentDataset = response.csv;
            currentTarget = response.target;
            displayDataPreview(response.csv);
            showSuccess('تم رفع البيانات بنجاح!');
        },
        error: function(xhr, status, error) {
            hideLoading($('#uploadData'));
            let errorMsg = 'فشل في الاتصال بالخادم';
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMsg = xhr.responseJSON.error;
            }
            showError('خطأ في رفع الملف: ' + errorMsg);
        }
    });
}

function fetchData() {
    const url = $('#dataUrl').val();
    if (!url) {
        showError('يرجى إدخال رابط البيانات');
        return;
    }
    
    showLoading($('#fetchData'));
    
    $.ajax({
        url: '/api/fetch',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ url: url }),
        success: function(response) {
            hideLoading($('#fetchData'));
            if (response.error) {
                showError('خطأ في جلب البيانات: ' + response.error);
                return;
            }
            
            currentDataset = response.csv;
            currentTarget = response.target;
            displayDataPreview(response.csv);
            showSuccess('تم جلب البيانات بنجاح!');
        },
        error: function(xhr, status, error) {
            hideLoading($('#fetchData'));
            let errorMsg = 'فشل في الاتصال بالخادم';
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMsg = xhr.responseJSON.error;
            }
            showError('خطأ في جلب البيانات: ' + errorMsg);
        }
    });
}

function displayDataPreview(csvData) {
    try {
        const rows = csvData.split('\n');
        const headers = rows[0].split(',');
        const sampleRows = rows.slice(1, 6);
        
        let headerHtml = '';
        headers.forEach(header => {
            headerHtml += `<th>${header}</th>`;
        });
        $('#previewHeader').html(headerHtml);
        
        let bodyHtml = '';
        sampleRows.forEach(row => {
            if (row.trim() === '') return;
            
            const cells = row.split(',');
            bodyHtml += '<tr>';
            cells.forEach(cell => {
                bodyHtml += `<td>${cell}</td>`;
            });
            bodyHtml += '</tr>';
        });
        $('#previewBody').html(bodyHtml);
        
        $('#dataInfo').html(`<i class="fas fa-info-circle me-1"></i> ${rows.length - 1} صف، ${headers.length} عمود`);
        $('#dataPreview').fadeIn();
    } catch (error) {
        console.error("❌ خطأ في عرض البيانات:", error);
        showError('خطأ في عرض معاينة البيانات');
    }
}

function runGeneticAlgorithm() {
    if (!currentDataset) {
        showError('يرجى تحميل أو توليد بيانات أولاً');
        return;
    }
    
    const popSize = $('#popSize').val();
    const generations = $('#generations').val();
    const crossoverRate = $('#crossoverRate').val();
    const mutationRate = $('#mutationRate').val();
    
    console.log(`🔄 تشغيل GA: popSize=${popSize}, generations=${generations}`);
    
    showLoading($('#runGA'));
    
    $.ajax({
        url: '/api/ga',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            df: currentDataset,
            target: currentTarget,
            pop_size: parseInt(popSize),
            generations: parseInt(generations),
            crossover_rate: parseFloat(crossoverRate),
            mutation_rate: parseFloat(mutationRate),
            use_cache: true
        }),
        success: function(response) {
            hideLoading($('#runGA'));
            if (response.error) {
                showError('خطأ في الخوارزمية الجينية: ' + response.error);
                return;
            }
            
            gaResult = response;
            displayGAResults(response);
            showSuccess('تم تشغيل الخوارزمية الجينية بنجاح!');
            checkCacheStatus();
            
            // عرض النتائج التفصيلية
            setTimeout(() => {
                displayGADetailedResults();
            }, 500);
            
            console.log("✅ تم تشغيل GA بنجاح");
        },
        error: function(xhr, status, error) {
            hideLoading($('#runGA'));
            let errorMsg = 'فشل في الاتصال بالخادم';
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMsg = xhr.responseJSON.error;
            }
            showError('خطأ في الخوارزمية الجينية: ' + errorMsg);
            console.error("❌ خطأ في GA:", error, xhr.responseJSON);
        }
    });
}

function displayGAResults(result) {
    $('#gaFinalScore').text(result.final_score ? result.final_score.toFixed(4) : '0.0000');
    $('#gaSelectedCount').text(result.selected_features ? result.selected_features.length : 0);
    
    let featuresHtml = '';
    if (result.selected_features && result.selected_features.length > 0) {
        result.selected_features.forEach(feature => {
            featuresHtml += `<div class="feature-item">${feature}</div>`;
        });
    } else {
        featuresHtml = '<p class="text-muted">لم يتم اختيار أي ميزات</p>';
    }
    $('#gaSelectedFeatures').html(featuresHtml);
    
    const cacheStatus = result.cached ? 'مخزنة' : 'جديدة';
    const cacheClass = result.cached ? 'bg-success' : 'bg-info';
    $('#gaCacheStatus, #gaCacheStatus2').text(cacheStatus)
        .removeClass('bg-secondary bg-danger bg-success bg-info')
        .addClass(cacheClass);
    
    $('#gaResults').fadeIn();
}

function displayGADetailedResults() {
    $.get('/api/results/ga')
        .done(function(response) {
            if (response.error) {
                console.error('خطأ في جلب نتائج GA:', response.error);
                return;
            }

            if (!response.ga_result) {
                $('#gaDetailedResultsContent').html(`
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i>
                        لا توجد نتائج مخزنة للخوارزمية الجينية
                    </div>
                `);
                return;
            }

            const ga = response.ga_result;
            let historyHtml = '';
            
            if (ga.history && ga.history.length > 0) {
                historyHtml = `
                    <h6>تطور اللياقة عبر الأجيال:</h6>
                    <div class="table-responsive">
                        <table class="table table-sm table-striped">
                            <thead>
                                <tr>
                                    <th>الجيل</th>
                                    <th>أفضل لياقة</th>
                                    <th>عدد الميزات</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${ga.history.map(gen => `
                                    <tr>
                                        <td>${gen.generation}</td>
                                        <td>${gen.best_fitness ? gen.best_fitness.toFixed(4) : 'N/A'}</td>
                                        <td>${gen.selected_count || 'N/A'}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                `;
            }

            const html = `
                <div class="row">
                    <div class="col-md-4">
                        <div class="ga-stat-card">
                            <div class="stat-icon bg-primary">
                                <i class="fas fa-star"></i>
                            </div>
                            <div class="stat-content">
                                <h3>${ga.final_score ? ga.final_score.toFixed(4) : 'N/A'}</h3>
                                <p>اللياقة النهائية</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="ga-stat-card">
                            <div class="stat-icon bg-success">
                                <i class="fas fa-list"></i>
                            </div>
                            <div class="stat-content">
                                <h3>${ga.selected_features ? ga.selected_features.length : 0}</h3>
                                <p>الميزات المختارة</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="ga-stat-card">
                            <div class="stat-icon bg-info">
                                <i class="fas fa-bolt"></i>
                            </div>
                            <div class="stat-content">
                                <h3>${ga.cached ? 'مخزنة' : 'جديدة'}</h3>
                                <p>حالة النتيجة</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mt-4">
                    <h6>الميزات المختارة:</h6>
                    <div class="feature-list">
                        ${ga.selected_features && ga.selected_features.length > 0 ? 
                            ga.selected_features.map(feature => `
                                <div class="feature-item">${feature}</div>
                            `).join('') : 
                            '<p class="text-muted">لم يتم اختيار أي ميزات</p>'
                        }
                    </div>
                </div>
                
                ${historyHtml}
            `;

            $('#gaDetailedResultsContent').html(html);
            $('#ga-results-detailed-section').fadeIn();
            
            // تمرير إلى قسم النتائج
            $('html, body').animate({
                scrollTop: $('#ga-results-detailed-section').offset().top - 70
            }, 800);
        })
        .fail(function() {
            console.error('فشل في جلب نتائج GA');
        });
}

function runAllTraditionalMethods() {
    if (!currentDataset) {
        showError('يرجى تحميل أو توليد بيانات أولاً');
        return;
    }
    
    console.log("🔄 تشغيل جميع الطرق التقليدية");
    
    showLoading($('#runTraditionalMethods'));
    
    $.ajax({
        url: '/api/run_all_traditional',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            df: currentDataset,
            target: currentTarget
        }),
        success: function(response) {
            hideLoading($('#runTraditionalMethods'));
            if (response.error) {
                showError('خطأ في الطرق التقليدية: ' + response.error);
                return;
            }
            
            // حفظ النتائج
            response.methods.forEach(method => {
                methodResults[method.method] = method;
            });
            
            showSuccess('تم تشغيل جميع الطرق التقليدية بنجاح!');
            checkCacheStatus();
            
            // عرض النتائج الفردية
            setTimeout(() => {
                displayTraditionalResults();
            }, 500);
            
            console.log("✅ تم تشغيل الطرق التقليدية بنجاح");
        },
        error: function(xhr, status, error) {
            hideLoading($('#runTraditionalMethods'));
            let errorMsg = 'فشل في الاتصال بالخادم';
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMsg = xhr.responseJSON.error;
            }
            showError('خطأ في الطرق التقليدية: ' + errorMsg);
            console.error("❌ خطأ في الطرق التقليدية:", error, xhr.responseJSON);
        }
    });
}

function displayTraditionalResults() {
    $.get('/api/results/traditional')
        .done(function(response) {
            if (response.error) {
                console.error('خطأ في جلب نتائج الطرق التقليدية:', response.error);
                return;
            }

            let html = '';
            response.methods.forEach(method => {
                const hasError = method.error || method.status === 'لم يتم التشغيل';
                const score = method.cv_score ? method.cv_score.toFixed(4) : 'غير متوفر';
                const featuresCount = method.selected_features ? method.selected_features.length : 0;
                const featuresList = method.selected_features ? method.selected_features.join(', ') : 'لا توجد';
                
                html += `
                    <div class="method-result-card ${hasError ? 'error-card' : 'success-card'}">
                        <div class="method-header">
                            <h6 class="method-name">${method.method}</h6>
                            <span class="badge ${hasError ? 'bg-danger' : 'bg-success'}">
                                ${hasError ? (method.error || 'لم يتم التشغيل') : 'مكتمل'}
                            </span>
                        </div>
                        <div class="method-details">
                            <div class="row">
                                <div class="col-md-3">
                                    <div class="stat-item">
                                        <strong>دقة التقييم:</strong>
                                        <span class="score">${score}</span>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="stat-item">
                                        <strong>عدد الميزات:</strong>
                                        <span class="count">${featuresCount}</span>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="stat-item">
                                        <strong>الميزات المختارة:</strong>
                                        <div class="features">${featuresList}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });

            $('#traditionalResultsContent').html(html);
            $('#traditional-results-section').fadeIn();
            
            // تمرير إلى قسم النتائج
            $('html, body').animate({
                scrollTop: $('#traditional-results-section').offset().top - 70
            }, 800);
        })
        .fail(function() {
            console.error('فشل في جلب نتائج الطرق التقليدية');
        });
}

function runAllStatisticalMethods() {
    if (!currentDataset) {
        showError('يرجى تحميل أو توليد بيانات أولاً');
        return;
    }
    
    console.log("🔄 تشغيل جميع الطرق الإحصائية");
    
    showLoading($('#runStatisticalMethods'));
    
    $.ajax({
        url: '/api/run_all_statistical',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            df: currentDataset,
            target: currentTarget
        }),
        success: function(response) {
            hideLoading($('#runStatisticalMethods'));
            if (response.error) {
                showError('خطأ في الطرق الإحصائية: ' + response.error);
                return;
            }
            
            // حفظ النتائج
            response.methods.forEach(method => {
                methodResults[method.method] = method;
            });
            
            showSuccess('تم تشغيل جميع الطرق الإحصائية بنجاح!');
            checkCacheStatus();
            
            // عرض النتائج الفردية
            setTimeout(() => {
                displayStatisticalResults();
            }, 500);
            
            console.log("✅ تم تشغيل الطرق الإحصائية بنجاح");
        },
        error: function(xhr, status, error) {
            hideLoading($('#runStatisticalMethods'));
            let errorMsg = 'فشل في الاتصال بالخادم';
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMsg = xhr.responseJSON.error;
            }
            showError('خطأ في الطرق الإحصائية: ' + errorMsg);
            console.error("❌ خطأ في الطرق الإحصائية:", error, xhr.responseJSON);
        }
    });
}

function displayStatisticalResults() {
    $.get('/api/results/statistical')
        .done(function(response) {
            if (response.error) {
                console.error('خطأ في جلب نتائج الطرق الإحصائية:', response.error);
                return;
            }

            let html = '';
            response.methods.forEach(method => {
                const hasError = method.error || method.status === 'لم يتم التشغيل';
                const score = method.cv_score ? method.cv_score.toFixed(4) : 'غير متوفر';
                const featuresCount = method.selected_features ? method.selected_features.length : 0;
                const featuresList = method.selected_features ? method.selected_features.join(', ') : 'لا توجد';
                
                html += `
                    <div class="method-result-card ${hasError ? 'error-card' : 'success-card'}">
                        <div class="method-header">
                            <h6 class="method-name">${method.method}</h6>
                            <span class="badge ${hasError ? 'bg-danger' : 'bg-success'}">
                                ${hasError ? (method.error || 'لم يتم التشغيل') : 'مكتمل'}
                            </span>
                        </div>
                        <div class="method-details">
                            <div class="row">
                                <div class="col-md-3">
                                    <div class="stat-item">
                                        <strong>دقة التقييم:</strong>
                                        <span class="score">${score}</span>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="stat-item">
                                        <strong>عدد الميزات:</strong>
                                        <span class="count">${featuresCount}</span>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="stat-item">
                                        <strong>الميزات المختارة:</strong>
                                        <div class="features">${featuresList}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });

            $('#statisticalResultsContent').html(html);
            $('#statistical-results-section').fadeIn();
            
            // تمرير إلى قسم النتائج
            $('html, body').animate({
                scrollTop: $('#statistical-results-section').offset().top - 70
            }, 800);
        })
        .fail(function() {
            console.error('فشل في جلب نتائج الطرق الإحصائية');
        });
}

function compareMethods() {
    if (!currentDataset) {
        showError('يرجى تحميل أو توليد بيانات أولاً');
        return;
    }
    
    const methods = Object.values(methodResults);
    
    if (gaResult) {
        methods.push(gaResult);
    }
    
    if (methods.length === 0) {
        showError('يرجى تشغيل طريقة واحدة على الأقل للمقارنة');
        return;
    }
    
    console.log(`🔄 بدء المقارنة بين ${methods.length} طريقة`);
    
    showLoading($('#compareMethods'));
    
    $.ajax({
        url: '/api/compare',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            df: currentDataset,
            target: currentTarget,
            methods: methods,
            include_ga_cached: true
        }),
        success: function(response) {
            hideLoading($('#compareMethods'));
            if (response.error) {
                showError('خطأ في المقارنة: ' + response.error);
                return;
            }
            
            displayComparisonResults(response);
            showSuccess('تمت المقارنة بنجاح!');
            console.log("✅ تمت المقارنة بنجاح");
        },
        error: function(xhr, status, error) {
            hideLoading($('#compareMethods'));
            let errorMsg = 'فشل في الاتصال بالخادم';
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMsg = xhr.responseJSON.error;
            }
            showError('خطأ في المقارنة: ' + errorMsg);
            console.error("❌ خطأ في المقارنة:", error, xhr.responseJSON);
        }
    });
}

function displayComparisonResults(results) {
    if (results.plots && results.plots.comparison_bar) {
        $('#comparisonChart').attr('src', `data:image/png;base64,${results.plots.comparison_bar}`);
    }
    
    if (results.plots && results.plots.ga_history) {
        $('#gaHistoryChart').attr('src', `data:image/png;base64,${results.plots.ga_history}`);
        $('#gaHistoryChart').parent().parent().show();
    } else {
        $('#gaHistoryChart').parent().parent().hide();
    }
    
    let resultsHtml = '';
    if (results.stats) {
        Object.keys(results.stats).forEach(method => {
            const stat = results.stats[method];
            resultsHtml += `
                <tr>
                    <td>${method}</td>
                    <td>${stat.fitness_score ? stat.fitness_score.toFixed(4) : 'غير متوفر'}</td>
                    <td>${stat.n_features}</td>
                    <td>${stat.selected_features ? stat.selected_features.join(', ') : 'لا توجد'}</td>
                </tr>
            `;
        });
    }
    $('#resultsBody').html(resultsHtml);
    
    $('#comparisonResults').fadeIn();
}

function clearAllCache() {
    $.ajax({
        url: '/api/cache/clear_all',
        type: 'POST',
        success: function(response) {
            if (response.success) {
                methodResults = {};
                gaResult = null;
                $('#gaResults').hide();
                $('#comparisonResults').hide();
                $('#traditional-results-section').hide();
                $('#statistical-results-section').hide();
                $('#ga-results-detailed-section').hide();
                showSuccess(response.message);
                checkCacheStatus();
            }
        },
        error: function(xhr, status, error) {
            let errorMsg = 'فشل في الاتصال بالخادم';
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMsg = xhr.responseJSON.error;
            }
            showError('فشل في مسح الذاكرة المؤقتة: ' + errorMsg);
        }
    });
}

function showLoading(element) {
    element.addClass('loading').prop('disabled', true);
    const originalText = element.html();
    element.data('original-text', originalText);
    element.html(`<i class="fas fa-spinner fa-spin me-2"></i> جاري المعالجة...`);
}

function hideLoading(element) {
    element.removeClass('loading').prop('disabled', false);
    const originalText = element.data('original-text');
    if (originalText) {
        element.html(originalText);
    }
}

function showError(message) {
    // استخدام alert بسيط للخطأ
    alert(`❌ ${message}`);
    console.error("❌ خطأ:", message);
}

function showSuccess(message) {
    alert(`✅ ${message}`);
    console.log("✅ نجاح:", message);
}