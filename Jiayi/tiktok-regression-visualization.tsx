import React, { useState, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Line, LineChart, ResponsiveContainer, Label } from 'recharts';

const TikTokRegressionAnalysis = () => {
  const [data, setData] = useState([]);
  const [regressionData, setRegressionData] = useState({});
  const [selectedRelationship, setSelectedRelationship] = useState('观看数 vs 点赞数');
  const [loading, setLoading] = useState(true);
  const [sampleSize, setSampleSize] = useState(500);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        // 读取CSV文件
        const fileContent = await window.fs.readFile('tiktok_dataset.csv', { encoding: 'utf-8' });
        
        // 使用Papaparse解析CSV
        const Papa = await import('papaparse');
        const parsedData = Papa.default.parse(fileContent, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true
        });
        
        // 过滤有效数据
        const validData = parsedData.data.filter(row => 
          row.video_view_count !== null && 
          row.video_like_count !== null &&
          row.video_share_count !== null &&
          row.video_download_count !== null &&
          row.video_comment_count !== null &&
          !isNaN(row.video_view_count) &&
          !isNaN(row.video_like_count) &&
          !isNaN(row.video_share_count) &&
          !isNaN(row.video_download_count) &&
          !isNaN(row.video_comment_count)
        );
        
        // 创建回归模型
        const relationships = [
          { x: 'video_view_count', y: 'video_like_count', label: '观看数 vs 点赞数' },
          { x: 'video_view_count', y: 'video_share_count', label: '观看数 vs 分享数' },
          { x: 'video_view_count', y: 'video_download_count', label: '观看数 vs 下载数' },
          { x: 'video_view_count', y: 'video_comment_count', label: '观看数 vs 评论数' },
          { x: 'video_like_count', y: 'video_comment_count', label: '点赞数 vs 评论数' }
        ];
        
        // 计算回归并储存结果
        const results = {};
        for (const rel of relationships) {
          const x = validData.map(row => row[rel.x]);
          const y = validData.map(row => row[rel.y]);
          const result = linearRegression(x, y);
          
          // 计算回归线
          const xMin = Math.min(...x);
          const xMax = Math.max(...x);
          const yMin = result.slope * xMin + result.intercept;
          const yMax = result.slope * xMax + result.intercept;
          
          results[rel.label] = {
            ...result,
            xName: rel.x,
            yName: rel.y,
            regressionLine: [
              { x: xMin, y: yMin },
              { x: xMax, y: yMax }
            ]
          };
        }
        
        // 随机抽样
        const sample = getRandomSample(validData, sampleSize);
        
        setData(validData);
        setRegressionData(results);
        setLoading(false);
      } catch (error) {
        console.error("Error loading data:", error);
        setLoading(false);
      }
    };
    
    fetchData();
  }, [sampleSize]);
  
  // 简单线性回归函数
  const linearRegression = (x, y) => {
    const n = x.length;
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumXX = 0;
    
    for (let i = 0; i < n; i++) {
      sumX += x[i];
      sumY += y[i];
      sumXY += x[i] * y[i];
      sumXX += x[i] * x[i];
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    // 计算R方
    let meanY = sumY / n;
    let totalSS = 0;
    let residualSS = 0;
    
    for (let i = 0; i < n; i++) {
      totalSS += Math.pow(y[i] - meanY, 2);
      residualSS += Math.pow(y[i] - (slope * x[i] + intercept), 2);
    }
    
    const rSquared = 1 - (residualSS / totalSS);
    
    return { slope, intercept, rSquared };
  };
  
  // 随机抽样函数
  const getRandomSample = (data, size) => {
    const sample = [];
    const dataClone = [...data];
    const sampleSize = Math.min(size, dataClone.length);
    
    for (let i = 0; i < sampleSize; i++) {
      const randomIndex = Math.floor(Math.random() * dataClone.length);
      sample.push(dataClone[randomIndex]);
      dataClone.splice(randomIndex, 1);
    }
    
    return sample;
  };
  
  // 当没有数据时显示加载信息
  if (loading) {
    return <div className="flex justify-center items-center h-64">正在加载数据...</div>;
  }
  
  // 准备当前选择的关系的数据
  const currentRelation = regressionData[selectedRelationship];
  const xName = currentRelation?.xName;
  const yName = currentRelation?.yName;
  
  // 获取随机样本
  const sample = getRandomSample(data, sampleSize);
  
  // 准备散点图数据
  const scatterData = sample.map(row => ({
    x: row[xName],
    y: row[yName]
  }));
  
  // 准备回归线数据
  const lineData = currentRelation?.regressionLine || [];
  
  // 格式化轴标签
  const formatAxisName = (name) => {
    switch(name) {
      case 'video_view_count': return '观看数';
      case 'video_like_count': return '点赞数';
      case 'video_share_count': return '分享数';
      case 'video_download_count': return '下载数';
      case 'video_comment_count': return '评论数';
      default: return name;
    }
  };
  
  return (
    <div className="flex flex-col p-4">
      <h1 className="text-2xl font-bold mb-4">TikTok数据线性回归分析</h1>
      
      <div className="mb-4">
        <label className="mr-2">选择关系:</label>
        <select 
          value={selectedRelationship}
          onChange={(e) => setSelectedRelationship(e.target.value)}
          className="border p-1 rounded"
        >
          {Object.keys(regressionData).map(key => (
            <option key={key} value={key}>{key}</option>
          ))}
        </select>
        
        <label className="ml-4 mr-2">样本大小:</label>
        <select 
          value={sampleSize}
          onChange={(e) => setSampleSize(parseInt(e.target.value))}
          className="border p-1 rounded"
        >
          <option value={100}>100</option>
          <option value={200}>200</option>
          <option value={500}>500</option>
          <option value={1000}>1000</option>
        </select>
      </div>
      
      {currentRelation && (
        <div className="bg-gray-100 p-4 rounded mb-4">
          <h2 className="text-xl font-bold mb-2">{selectedRelationship}的回归分析结果</h2>
          <p><strong>回归公式:</strong> {formatAxisName(yName)} = {currentRelation.slope.toFixed(4)} × {formatAxisName(xName)} + {currentRelation.intercept.toFixed(4)}</p>
          <p><strong>决定系数 (R²):</strong> {currentRelation.rSquared.toFixed(4)}</p>
          <p><strong>相关性强度:</strong> {
            currentRelation.rSquared > 0.7 ? '强相关' : 
            currentRelation.rSquared > 0.4 ? '中等相关' : '弱相关'
          }</p>
        </div>
      )}
      
      <div className="h-96 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
          >
            <CartesianGrid />
            <XAxis 
              type="number" 
              dataKey="x" 
              name={formatAxisName(xName)}
              label={{ value: formatAxisName(xName), position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              type="number" 
              dataKey="y" 
              name={formatAxisName(yName)}
              label={{ value: formatAxisName(yName), angle: -90, position: 'insideLeft' }}
            />
            <Tooltip 
              formatter={(value) => Math.round(value)}
              labelFormatter={(value) => `${formatAxisName(xName)}: ${Math.round(value)}`}
            />
            <Legend />
            <Scatter 
              name="数据点" 
              data={scatterData} 
              fill="#8884d8" 
              shape="circle"
              opacity={0.5}
            />
            
            {/* 回归线 */}
            <Scatter
              name="回归线"
              data={lineData}
              line={{ stroke: '#ff7300', strokeWidth: 2 }}
              shape={() => null}
              legendType="line"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TikTokRegressionAnalysis;
