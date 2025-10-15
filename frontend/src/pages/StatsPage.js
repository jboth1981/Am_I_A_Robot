import React, { useEffect, useMemo, useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { useAuth } from '../contexts/AuthContext';
import { authService } from '../services/authService';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const StatsPage = () => {
  const { token, isGuest, user } = useAuth();
  const [submissions, setSubmissions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    const load = async () => {
      if (!token || isGuest) {
        setLoading(false);
        return;
      }
      try {
        const data = await authService.getUserSubmissions(token, 50);
        if (!active) return;
        setSubmissions(data || []);
      } catch (e) {
        if (!active) return;
        setError(e.message || 'Failed to load submissions');
      } finally {
        if (active) setLoading(false);
      }
    };
    load();
    return () => { active = false; };
  }, [token, isGuest]);

  const chartData = useMemo(() => {
    if (!submissions || submissions.length === 0) return null;
    const labels = submissions
      .slice()
      .reverse()
      .map(s => new Date(s.completed_at).toLocaleString());
    const unpredictability = submissions
      .slice()
      .reverse()
      .map(s => 1 - (s.accuracy_percentage / 100));

    return {
      labels,
      datasets: [
        {
          label: 'Unpredictability (1 - accuracy)',
          data: unpredictability,
          borderColor: 'rgba(99, 102, 241, 1)',
          backgroundColor: 'rgba(99, 102, 241, 0.2)',
          tension: 0.25,
          pointRadius: 3
        }
      ]
    };
  }, [submissions]);

  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Your Unpredictability Over Time' }
    },
    scales: {
      y: {
        beginAtZero: true,
        suggestedMax: 1,
        ticks: {
          callback: (value) => `${Math.round(value * 100)}%`
        }
      }
    }
  };

  if (isGuest || !user) {
    return (
      <div className="card">
        <h2>Stats</h2>
        <p>Please log in to view your historical stats.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="card">Loading your stats…</div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <h2>Stats</h2>
        <p style={{ color: 'crimson' }}>{error}</p>
      </div>
    );
  }

  if (!chartData) {
    return (
      <div className="card">
        <h2>Stats</h2>
        <p>No submissions yet. Complete a game to see your stats here.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>Stats</h2>
      <div style={{ maxWidth: '900px' }}>
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
};

export default StatsPage;


