import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { HashRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import {
  AppBar, Toolbar, Typography, Container, TextField, Button, Box,
  Card, CardMedia, CardContent, CircularProgress, Paper, CssBaseline,
  ThemeProvider, createTheme, List, ListItem, Avatar, Grid, GlobalStyles,
  Dialog, DialogContent, IconButton, ListItemText, Chip // MODIFIED: Added Chip for the modal
} from '@mui/material';
import { Bar, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement } from 'chart.js';
import SendIcon from '@mui/icons-material/Send';
import SmartToyOutlinedIcon from '@mui/icons-material/SmartToyOutlined';
import PersonOutlineOutlinedIcon from '@mui/icons-material/PersonOutlineOutlined';
import InsightsIcon from '@mui/icons-material/Insights';
import DonutLargeIcon from '@mui/icons-material/DonutLarge';
import AttachMoneyIcon from '@mui/icons-material/AttachMoney';
import RemoveIcon from '@mui/icons-material/Remove';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import CloseIcon from '@mui/icons-material/Close';
import Inventory2OutlinedIcon from '@mui/icons-material/Inventory2Outlined';
import StorefrontOutlinedIcon from '@mui/icons-material/StorefrontOutlined';
import TrendingDownOutlinedIcon from '@mui/icons-material/TrendingDownOutlined';

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend);

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#6366f1', light: '#818cf8' },
    secondary: { main: '#ec4899' },
    background: { default: '#f8fafc', paper: '#ffffff' },
    text: { primary: '#1e293b', secondary: '#64748b' },
  },
  typography: {
    fontFamily: "'Inter', 'SF Pro Display', -apple-system, sans-serif",
    h3: { fontWeight: 700, letterSpacing: '-0.02em' },
    h4: { fontWeight: 600, letterSpacing: '-0.01em' },
    h5: { fontWeight: 600 },
    body1: { lineHeight: 1.6 },
  },
  shape: { borderRadius: 16 },
});

const API_BASE_URL = 'http://127.0.0.1:8000';

const formatPrice = (price) => {
  if (price === null || price === undefined || price === 'N/A' || price === 'nan' || price === '') {
    return 'Price not available';
  }
  const priceAsString = String(price);
  const cleanPrice = priceAsString.replace(/[$,]/g, '');
  const numValue = parseFloat(cleanPrice);
  if (isNaN(numValue)) {
    return 'Price not available';
  }
  return `$${numValue.toFixed(2)}`;
};

const glassStyles = {
  background: 'rgba(255, 255, 255, 0.7)',
  backdropFilter: 'blur(20px) saturate(180%)',
  WebkitBackdropFilter: 'blur(20px) saturate(180%)',
  border: '1px solid rgba(255, 255, 255, 0.3)',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
};

function AppHeader() {
  const location = useLocation();
  return (
    <AppBar
      position="fixed"
      elevation={0}
      sx={{
        ...glassStyles,
        borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
      }}
    >
      <Container maxWidth={false} sx={{ maxWidth: '2560px' }}>
        <Toolbar sx={{ py: 1.5, px: 4 }}>
          <Typography
            variant="h4"
            component="div"
            sx={{
              flexGrow: 1,
              fontWeight: 700,
              background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Ikarus
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            {[['/', 'RECOMMENDATIONS'], ['/analytics', 'ANALYTICS']].map(([path, label]) => (
              <Link key={path} to={path} style={{ textDecoration: 'none' }}>
                <Button
                  variant={location.pathname === path ? 'contained' : 'text'}
                  sx={{
                    px: 4,
                    py: 1.5,
                    fontSize: '0.95rem',
                    fontWeight: 600,
                    letterSpacing: '0.5px',
                    color: location.pathname === path ? 'white' : 'text.primary',
                    background: location.pathname === path
                      ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
                      : 'transparent',
                    '&:hover': {
                      background: location.pathname === path
                        ? 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)'
                        : 'rgba(99, 102, 241, 0.1)',
                    },
                  }}
                >
                  {label}
                </Button>
              </Link>
            ))}
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
}

function ProductCard({ product, onClick }) {
  return (
    <Card
      onClick={onClick}
      sx={{
        cursor: 'pointer',
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        ...glassStyles,
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:hover': {
          transform: 'translateY(-8px)',
          boxShadow: '0 20px 60px rgba(99, 102, 241, 0.2)',
          border: '1px solid rgba(99, 102, 241, 0.4)',
        },
      }}
    >
      <CardMedia
        component="img"
        sx={{
          height: 220,
          objectFit: 'cover',
          flexShrink: 0,
          background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%)',
        }}
        image={product.image_url}
        alt={product.title}
        onError={(e) => {
          e.target.onerror = null;
          e.target.src = "https://placehold.co/400x220/f8fafc/6366f1?text=No+Image";
        }}
      />
      <CardContent sx={{ p: 3, flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
        <Box>
          <Typography
            variant="h6"
            sx={{
              mb: 2,
              fontWeight: 600,
              fontSize: '1rem',
              color: 'text.primary',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              minHeight: '3em',
            }}
          >
            {product.title}
          </Typography>
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{
              mb: 2,
              fontSize: '0.875rem',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              minHeight: '2.6em',
            }}
          >
            {product.generated_description}
          </Typography>
        </Box>
        <Typography
          sx={{
            fontSize: '1.5rem',
            fontWeight: 700,
            background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          {formatPrice(product.price)}
        </Typography>
      </CardContent>
    </Card>
  );
}


// ===================================================================
// ## MODIFIED: This is the newly integrated ProductModal component ##
// ===================================================================
function ProductModal({ product, open, onClose }) {
  if (!product) return null;
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          ...glassStyles,
          maxWidth: '1200px',
          borderRadius: '24px',
        }
      }}
    >
      <IconButton
        onClick={onClose}
        sx={{
          position: 'absolute',
          right: 16,
          top: 16,
          color: 'text.secondary',
          background: 'rgba(255, 255, 255, 0.9)',
          '&:hover': { background: 'rgba(255, 255, 255, 1)' },
          zIndex: 1,
        }}
      >
        <CloseIcon />
      </IconButton>
      <DialogContent sx={{ p: { xs: 3, md: 6 } }}>
        <Grid container spacing={5}>
          <Grid item xs={12} md={6}>
            <CardMedia
              component="img"
              sx={{
                width: '100%',
                height: 500,
                objectFit: 'cover',
                borderRadius: '16px',
                background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%)',
              }}
              image={product.image_url}
              alt={product.title}
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = "https://placehold.co/500x500/f8fafc/6366f1?text=No+Image";
              }}
            />
          </Grid>
          <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h4" sx={{ fontWeight: 600, mb: 2, color: 'text.primary' }}>
              {product.title}
            </Typography>
            <Typography
              variant="h4"
              sx={{
                mb: 3,
                fontWeight: 700,
                background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              {formatPrice(product.price)}
            </Typography>

            {/* NEW: Displaying the extra details */}
            <Box sx={{ mb: 3, display: 'flex', flexWrap: 'wrap', gap: 1.5 }}>
              {product.material && product.material !== 'Unknown' && <Chip label={`Material: ${product.material}`} variant="outlined" color="primary" />}
              {product.country_of_origin && product.country_of_origin !== 'Unknown' && <Chip label={`Origin: ${product.country_of_origin}`} variant="outlined" color="primary" />}
              {product.dimensions && product.dimensions !== 'N/a' && product.dimensions !== 'unknown' && <Chip label={`Dimensions: ${product.dimensions}`} variant="outlined" color="primary" />}
            </Box>

            <Typography variant="body1" sx={{ fontStyle: 'italic', color: 'text.secondary', mb: 2, lineHeight: 1.7 }}>
              "{product.generated_description || 'No detailed description available.'}"
            </Typography>

          </Grid>
        </Grid>
      </DialogContent>
    </Dialog>
  );
}


function RecommendationPage() {
  const [messages, setMessages] = useState([
    { text: "Welcome! How can I assist with your furniture search today?", sender: 'bot' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const messagesEndRef = useRef(null);
  const latestRecommendationRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    if (latestRecommendationRef.current && !isLoading) {
      setTimeout(() => {
        latestRecommendationRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    }
  }, [messages, isLoading]);

  const handleSendMessage = async (event) => {
    event.preventDefault();
    if (!inputValue.trim()) return;

    const newMessages = [...messages, { text: inputValue, sender: 'user' }];
    setMessages(newMessages);

    const query = inputValue;
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/recommend`, { query });
      const { products } = response.data;
      setMessages(prev => [...prev, {
        text: products.length > 0
          ? "Here are some perfect matches for you:"
          : "I couldn't find any products matching that. Try something else?",
        sender: 'bot',
        products,
        isLatest: true
      }]);
    } catch (error) {
      console.error("Error:", error);
      setMessages(prev => [...prev, {
        text: "Connection failed. Please ensure the backend is running.",
        sender: 'bot'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage(event);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        pt: '88px',
        background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 25%, #fef3c7 50%, #fce7f3 75%, #f0f9ff 100%)',
        backgroundSize: '400% 400%',
        animation: 'gradientShift 15s ease infinite',
        '@keyframes gradientShift': {
          '0%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
          '100%': { backgroundPosition: '0% 50%' },
        },
      }}
    >
      <Container maxWidth={false} sx={{ maxWidth: '2560px', py: 4, px: { xs: 2, sm: 4, md: 6 }, height: 'calc(100vh - 88px)', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ display: 'flex', height: '100%', gap: { xs: 2, md: 4 } }}>
          <Paper
            sx={{
              flex: { xs: '1 1 100%', md: '1 1 40%' },
              display: 'flex',
              flexDirection: 'column',
              ...glassStyles,
              borderRadius: '24px',
              overflow: 'hidden',
            }}
          >
            <Box sx={{ flexGrow: 1, overflowY: 'auto', p: { xs: 2, md: 4 } }}>
              <List>
                {messages.map((msg, index) => (
                  <ListItem key={index} sx={{ justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start', mb: 3, px: 0 }}>
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2, flexDirection: msg.sender === 'user' ? 'row-reverse' : 'row', maxWidth: '90%' }}>
                      <Avatar sx={{ background: msg.sender === 'bot' ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)' : 'linear-gradient(135deg, #ec4899 0%, #f43f5e 100%)', width: 44, height: 44, boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)' }}>
                        {msg.sender === 'bot' ? <SmartToyOutlinedIcon /> : <PersonOutlineOutlinedIcon />}
                      </Avatar>
                      <Paper sx={{ p: { xs: 2, md: 3 }, background: msg.sender === 'user' ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)' : 'rgba(255, 255, 255, 0.9)', borderRadius: msg.sender === 'user' ? '20px 20px 4px 20px' : '20px 20px 20px 4px', border: '1px solid rgba(148, 163, 184, 0.2)', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)' }}>
                        <Typography variant="body1" sx={{ color: 'text.primary' }}>{msg.text}</Typography>
                      </Paper>
                    </Box>
                  </ListItem>
                ))}
                {isLoading && (
                  <ListItem sx={{ justifyContent: 'flex-start', px: 0 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Avatar sx={{ background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', width: 44, height: 44 }}><SmartToyOutlinedIcon /></Avatar>
                      <Paper sx={{ p: { xs: 2, md: 3 }, ...glassStyles }}><CircularProgress size={24} sx={{ color: 'primary.main' }} /></Paper>
                    </Box>
                  </ListItem>
                )}
                <div ref={messagesEndRef} />
              </List>
            </Box>
            <Box component="form" onSubmit={handleSendMessage} sx={{ p: { xs: 2, md: 4 }, borderTop: '1px solid rgba(148, 163, 184, 0.2)', background: 'rgba(255, 255, 255, 0.5)' }}>
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-end' }}>
                <TextField fullWidth multiline maxRows={3} variant="outlined" placeholder="Describe the furniture you're looking for..." value={inputValue} onChange={(e) => setInputValue(e.target.value)} onKeyDown={handleKeyDown} disabled={isLoading} sx={{ '& .MuiOutlinedInput-root': { background: 'rgba(255, 255, 255, 0.9)', borderRadius: '16px', '& fieldset': { borderColor: 'rgba(148, 163, 184, 0.3)' }, '&:hover fieldset': { borderColor: 'rgba(99, 102, 241, 0.5)' }, '&.Mui-focused fieldset': { borderColor: 'primary.main' } } }} />
                <Button type="submit" variant="contained" disabled={isLoading} sx={{ minWidth: { xs: 'auto', sm: '120px' }, height: '56px', px: { xs: 2, sm: 4 }, background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', boxShadow: '0 4px 16px rgba(99, 102, 241, 0.3)', borderRadius: '16px', '&:hover': { background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)', boxShadow: '0 6px 24px rgba(99, 102, 241, 0.4)' } }}><SendIcon /></Button>
              </Box>
            </Box>
          </Paper>
          <Box sx={{ flex: { xs: '1 1 100%', md: '1 1 60%' }, overflowY: 'auto', pr: 1 }}>
            {messages.filter(m => m.products && m.products.length > 0).map((msg, msgIdx) => (
              <Box
                key={msgIdx}
                sx={{ mb: 4 }}
                ref={msg.isLatest ? latestRecommendationRef : null}
              >
                <Typography variant="h5" sx={{ mb: 3, fontWeight: 600, color: 'text.primary', minHeight: '2.5em', display: 'flex', alignItems: 'center' }}>Recommendations</Typography>
                <Grid container spacing={{ xs: 2, md: 3 }}>
                  {msg.products.map((p, idx) => (
                    <Grid item xs={12} sm={6} lg={4} key={p.id || idx}>
                      <ProductCard product={p} onClick={() => setSelectedProduct(p)} />
                    </Grid>
                  ))}
                </Grid>
              </Box>
            ))}
            {messages.every(m => !m.products || m.products.length === 0) && !isLoading && (
              <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', ...glassStyles, borderRadius: '24px', p: { xs: 4, md: 8 } }}>
                <Typography variant="h5" color="text.secondary" align="center">Your product recommendations will appear here</Typography>
              </Box>
            )}
          </Box>
        </Box>
        <ProductModal product={selectedProduct} open={!!selectedProduct} onClose={() => setSelectedProduct(null)} />
      </Container>
    </Box>
  );
}

function AnalyticsPage() {
  const [analyticsData, setAnalyticsData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/analytics`);
        setAnalyticsData(response.data);
        setError(null);
      } catch (err) {
        console.error(err);
        setError('Failed to fetch analytics data. Please check if the backend is running.');
      } finally {
        setIsLoading(false);
      }
    };
    fetchAnalytics();
  }, []);

  const StatCard = ({ title, value, icon, gradient }) => (
    <Paper sx={{ p: 3, height: '100%', ...glassStyles, background: `linear-gradient(135deg, ${gradient})`, transition: 'all 0.3s ease', '&:hover': { transform: 'translateY(-4px)', boxShadow: '0 12px 40px rgba(0, 0, 0, 0.12)' } }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: '100%' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body2" sx={{ textTransform: 'uppercase', letterSpacing: '0.1em', color: 'text.secondary', fontWeight: 600 }}>{title}</Typography>
          <Avatar sx={{ background: 'rgba(255, 255, 255, 0.3)', backdropFilter: 'blur(10px)', color: 'text.primary' }}>{icon}</Avatar>
        </Box>
        <Typography variant="h4" sx={{ fontWeight: 700, color: 'text.primary', mt: 2 }}>{value}</Typography>
      </Box>
    </Paper>
  );

  if (isLoading) {
    return (
      <Box sx={{ minHeight: '100vh', pt: '88px', background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <CircularProgress size={60} sx={{ color: 'primary.main' }} />
      </Box>
    );
  }

  if (error || !analyticsData) {
    return (
      <Box sx={{ minHeight: '100vh', pt: '88px', background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)', display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: 2 }}>
        <Typography color="error" variant="h6" align="center">{error || 'Failed to fetch analytics data'}</Typography>
        <Button variant="contained" onClick={() => window.location.reload()} sx={{ background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)' }}>
          Retry
        </Button>
      </Box>
    );
  }

  const topBrands = Object.entries(analyticsData.brand_counts || {}).slice(0, 10);
  const topCategories = Object.entries(analyticsData.category_counts || {}).slice(0, 10).map(([label, value]) => [label.replace(/[\[\]"]/g, ''), value]);
  const priceStats = analyticsData.price_stats || {};
  
  const totalProducts = analyticsData.total_rows || priceStats.count || 0;
  
  const minPriceValue = priceStats.min ? formatPrice(priceStats.min) : 'N/A';
  
  const categoryColors = ['#6366f1', '#8b5cf6', '#ec4899', '#f43f5e', '#f59e0b', '#10b981', '#14b8a6', '#06b6d4', '#3b82f6', '#a855f7'];

  const barChartOptions = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { backgroundColor: 'rgba(255, 255, 255, 0.95)', titleColor: '#1e293b', bodyColor: '#64748b', borderColor: 'rgba(99, 102, 241, 0.2)', borderWidth: 1, cornerRadius: 12, padding: 16 },
    },
    scales: {
      x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(148, 163, 184, 0.15)' } },
      y: { ticks: { color: '#64748b', font: { size: 12 } }, grid: { display: false } },
    },
    layout: { padding: { top: 10, bottom: 10 } }
  };

  const brandChartData = {
    labels: topBrands.map(([label]) => label),
    datasets: [{ label: '# of Products', data: topBrands.map(([, value]) => value), backgroundColor: 'rgba(99, 102, 241, 0.8)', borderColor: '#6366f1', borderWidth: 2, borderRadius: 8 }],
  };

  const pieChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { backgroundColor: 'rgba(255, 255, 255, 0.95)', titleColor: '#1e293b', bodyColor: '#64748b', borderColor: 'rgba(99, 102, 241, 0.2)', borderWidth: 1, cornerRadius: 12, padding: 16 },
    }
  };

  const categoryChartData = {
    labels: topCategories.map(([label]) => label),
    datasets: [{ data: topCategories.map(([, value]) => value), backgroundColor: categoryColors, borderWidth: 3, borderColor: '#ffffff' }],
  };

  return (
    <Box sx={{ minHeight: '100vh', pt: '88px', pb: 4, background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 25%, #fef3c7 50%, #fce7f3 75%, #f0f9ff 100%)', backgroundSize: '400% 400%', animation: 'gradientShift 15s ease infinite' }}>
      <Container maxWidth={false} sx={{ maxWidth: '2560px', px: { xs: 2, sm: 4, md: 6 } }}>
        <Typography variant="h3" align="center" sx={{ mb: { xs: 3, md: 5 }, pt: 2, fontWeight: 700, background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Analytics Dashboard
        </Typography>

        <Grid container spacing={{ xs: 2, md: 4 }}>
          <Grid item lg={2} md={4} sm={6} xs={12}>
            <StatCard title="Total Products" value={totalProducts.toLocaleString()} icon={<Inventory2OutlinedIcon />} gradient="rgba(16, 185, 129, 0.1) 0%, rgba(20, 184, 166, 0.1) 100%" />
          </Grid>
          <Grid item lg={2} md={4} sm={6} xs={12}>
            <StatCard title="Unique Brands" value={Object.keys(analyticsData.brand_counts || {}).length.toLocaleString()} icon={<StorefrontOutlinedIcon />} gradient="rgba(14, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%" />
          </Grid>
          <Grid item lg={2} md={4} sm={6} xs={12}>
            <StatCard title="Mean Price" value={formatPrice(priceStats.mean)} icon={<AttachMoneyIcon />} gradient="rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%" />
          </Grid>
          <Grid item lg={2} md={4} sm={6} xs={12}>
            <StatCard title="Median Price" value={formatPrice(priceStats.median)} icon={<RemoveIcon />} gradient="rgba(236, 72, 153, 0.1) 0%, rgba(244, 63, 94, 0.1) 100%" />
          </Grid>
          <Grid item lg={2} md={4} sm={6} xs={12}>
            <StatCard title="Max Price" value={formatPrice(priceStats.max)} icon={<TrendingUpIcon />} gradient="rgba(245, 158, 11, 0.1) 0%, rgba(239, 68, 68, 0.1) 100%" />
          </Grid>
          <Grid item lg={2} md={4} sm={6} xs={12}>
            <StatCard title="Min Price" value={minPriceValue} icon={<TrendingDownOutlinedIcon />} gradient="rgba(139, 92, 246, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%" />
          </Grid>
          
          <Grid item xs={12} lg={7}>
            <Paper sx={{ p: { xs: 3, md: 4 }, height: '100%', minHeight: '500px', ...glassStyles, borderRadius: '24px', display: 'flex', flexDirection: 'column' }}>
              <Typography variant="h5" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 2, color: 'text.primary', mb: 3 }}>
                <InsightsIcon color="primary" /> Top 10 Brands
              </Typography>
              {topBrands.length > 0 ? (
                <Box sx={{ flexGrow: 1, position: 'relative' }}>
                  <Bar data={brandChartData} options={barChartOptions} />
                </Box>
              ) : (
                <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Typography color="text.secondary">No brand data available</Typography>
                </Box>
              )}
            </Paper>
          </Grid>
          
          <Grid item xs={12} lg={5}>
            <Grid container spacing={{ xs: 2, md: 4 }} sx={{ height: '100%' }}>
              <Grid item xs={12}>
                <Paper sx={{ p: { xs: 3, md: 4 }, ...glassStyles, borderRadius: '24px' }}>
                  <Typography variant="h5" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 2, color: 'text.primary', mb: 2 }}>
                    <DonutLargeIcon color="primary" /> Top 10 Categories
                  </Typography>
                  {topCategories.length > 0 ? (
                    <Box sx={{ height: '250px', position: 'relative' }}>
                      <Pie data={categoryChartData} options={pieChartOptions} />
                    </Box>
                  ) : (
                    <Box sx={{ height: '250px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Typography color="text.secondary">No category data available</Typography>
                    </Box>
                  )}
                </Paper>
              </Grid>
              <Grid item xs={12}>
                <Paper sx={{ p: { xs: 2, md: 3 }, ...glassStyles, borderRadius: '24px', flexGrow: 1 }}>
                  <List dense>
                    {topCategories.map(([label, value], index) => (
                      <ListItem key={label} disableGutters>
                        <Box sx={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: categoryColors[index % categoryColors.length], mr: 2, flexShrink: 0 }} />
                        <ListItemText primary={label} primaryTypographyProps={{ variant: 'body2', noWrap: true, title: label, color: 'text.secondary' }} />
                        <Typography variant="body2" sx={{ fontWeight: 600, ml: 2 }}>{value}</Typography>
                      </ListItem>
                    ))}
                    {topCategories.length === 0 && (
                      <ListItem disableGutters>
                        <Typography color="text.secondary">No categories available</Typography>
                      </ListItem>
                    )}
                  </List>
                </Paper>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <GlobalStyles styles={{
        body: { backgroundColor: '#f8fafc', margin: 0, padding: 0, overflow: 'hidden', width: '100vw', height: '100vh' },
        '#root': { minHeight: '100vh', width: '100%' },
        '::-webkit-scrollbar': { width: '12px', height: '12px' },
        '::-webkit-scrollbar-track': { background: 'rgba(248, 250, 252, 0.5)', borderRadius: '6px' },
        '::-webkit-scrollbar-thumb': { background: 'rgba(99, 102, 241, 0.3)', borderRadius: '6px', '&:hover': { background: 'rgba(99, 102, 241, 0.5)' } },
        '*': { scrollbarWidth: 'thin', scrollbarColor: 'rgba(99, 102, 241, 0.3) rgba(248, 250, 252, 0.5)' },
        '@keyframes gradientShift': {
          '0%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
          '100%': { backgroundPosition: '0% 50%' },
        }
      }} />
      <Router>
        <AppHeader />
        <Routes>
          <Route path="/" element={<RecommendationPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;