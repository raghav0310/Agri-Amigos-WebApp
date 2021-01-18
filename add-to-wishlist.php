<!DOCTYPE HTML>
<html>
	<head>
	<meta charset="utf-8">
	<meta http-equiv="X-UA-Compatible" content="IE=edge">
	<title>Store Template</title>
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<meta name="description" content="" />
	<meta name="keywords" content="" />
	<meta name="author" content="" />

  <!-- Facebook and Twitter integration -->
	<meta property="og:title" content=""/>
	<meta property="og:image" content=""/>
	<meta property="og:url" content=""/>
	<meta property="og:site_name" content=""/>
	<meta property="og:description" content=""/>
	<meta name="twitter:title" content="" />
	<meta name="twitter:image" content="" />
	<meta name="twitter:url" content="" />
	<meta name="twitter:card" content="" />

	<link href="https://fonts.googleapis.com/css?family=Roboto:100,300,400,500,700" rel="stylesheet">
	
	<!-- Animate.css -->
	<link rel="stylesheet" href="css/animate.css">
	<!-- Icomoon Icon Fonts-->
	<link rel="stylesheet" href="css/icomoon.css">
	<!-- Bootstrap  -->
	<link rel="stylesheet" href="css/bootstrap.css">

	<!-- Magnific Popup -->
	<link rel="stylesheet" href="css/magnific-popup.css">

	<!-- Flexslider  -->
	<link rel="stylesheet" href="css/flexslider.css">

	<!-- Owl Carousel -->
	<link rel="stylesheet" href="css/owl.carousel.min.css">
	<link rel="stylesheet" href="css/owl.theme.default.min.css">
	
	<!-- Date Picker -->
	<link rel="stylesheet" href="css/bootstrap-datepicker.css">
	<!-- Flaticons  -->
	<link rel="stylesheet" href="fonts/flaticon/font/flaticon.css">

	<!-- Theme style  -->
	<link rel="stylesheet" href="css/style.css">

	<!-- Modernizr JS -->
	<script src="js/modernizr-2.6.2.min.js"></script>
	<!-- FOR IE9 below -->
	<!--[if lt IE 9]>
	<script src="js/respond.min.js"></script>
	<![endif]-->

	</head>
	<body>
	<?php
		$servername="localhost";
		$user="root";
		$psw="";
		$dbname="test";
		$conn=mysqli_connect($servername,$user,$psw,$dbname);
		if(!$conn)
		{
			die("connection failed:".mysqli_connect_error());
		}
		try{
			$sql="SELECT * FROM ADMINLOST";
			$result=$conn->query($sql);
			echo "<table border='1' cellpadding='5' width='95%' ALIGN='CENTER'>";
			echo "<tr>";
				echo "<th>Product Name</th>";
				echo "<th>Product Model</th>";
				echo "<th>Product Color</th>";
				echo "<th>Student Name</th>";
				echo "<th>Student Id</th>";
				echo "<th>Student Mobile</th>";
				echo "<th>Product Description</th>";
				echo "<th>Lost / Found </th>";
			echo "</tr>";
			while($row=$result->fetch_assoc())
			{
				echo "<tr>";
					echo "<td>".$row["p_name"]."</td>";
					echo "<td>".$row["p_model"]."</td>";
					echo "<td>".$row["p_color"]."</td>";
					echo "<td>".$row["s_name"]."</td>";
					echo "<td>".$row["s_id"]."</td>";
					echo "<td>".$row["s_mob"]."</td>";
					echo "<td>".$row["p_desc"]."</td>";
					echo "<td>".$row["lofo"]."</td>";
				echo "</tr>";
			}
			echo "</table>";
		}
		
	<div class="colorlib-loader"></div>

	<div id="page">
		<nav class="colorlib-nav" role="navigation">
			<div class="top-menu">
				<div class="container">
					<div class="row">
						<div class="col-xs-2">
							<div id="colorlib-logo"><a href="index.html">Store</a></div>
						</div>
						<div class="col-xs-10 text-right menu-1">
							<ul>
								<li><a href="index.html">Home</a></li>
								<li class="has-dropdown active">
									<a href="shop.html">Shop</a>
									<ul class="dropdown">
										<li><a href="product-detail.html">Product Detail</a></li>
										<li><a href="cart.html">Shipping Cart</a></li>
										<li><a href="checkout.html">Checkout</a></li>
										<li><a href="order-complete.html">Order Complete</a></li>
										<li><a href="add-to-wishlist.html">Wishlist</a></li>
									</ul>
								</li>
								<li><a href="blog.html">Blog</a></li>
								<li><a href="about.html">About</a></li>
								<li><a href="contact.html">Contact</a></li>
								<li><a href="cart.html"><i class="icon-shopping-cart"></i> Cart [0]</a></li>
							</ul>
						</div>
					</div>
				</div>
			</div>
		</nav>
		<aside id="colorlib-hero" class="breadcrumbs">
			<div class="flexslider">
				<ul class="slides">
			   	<li style="background-image: url(images/cover-img-1.jpg);">
			   		<div class="overlay"></div>
			   		<div class="container-fluid">
			   			<div class="row">
				   			<div class="col-md-6 col-md-offset-3 col-sm-12 col-xs-12 slider-text">
				   				<div class="slider-text-inner text-center">
				   					<h1>Wishlist</h1>
				   					<h2 class="bread"><span><a href="index.html">Home</a></span> <span>Wishlist</span></h2>
				   				</div>
				   			</div>
				   		</div>
			   		</div>
			   	</li>
			  	</ul>
		  	</div>
		</aside>

		<div class="colorlib-shop">
			<div class="container">
				<div class="row row-pb-md">
					<div class="col-md-10 col-md-offset-1">
						<div class="product-name">
							<div class="one-forth text-center">
								<span>Product Details</span>
							</div>
							<div class="one-eight text-center">
								<span>Price</span>
							</div>
							<div class="one-eight text-center">
								<span>Quantity</span>
							</div>
							<div class="one-eight text-center">
								<span>Total</span>
							</div>
							<div class="one-eight text-center">
								<span>Remove</span>
							</div>
						</div>
						<div class="product-cart">
							<div class="one-forth">
								<div class="product-img" style="background-image: url(images/item-6.jpg);">
								</div>
								<div class="display-tc">
									<h3>Product Name</h3>
								</div>
							</div>
							<div class="one-eight text-center">
								<div class="display-tc">
									<span class="price">$68.00</span>
								</div>
							</div>
							<div class="one-eight text-center">
								<div class="display-tc">
									<input type="text" id="quantity" name="quantity" class="form-control input-number text-center" value="1" min="1" max="100">
								</div>
							</div>
							<div class="one-eight text-center">
								<div class="display-tc">
									<span class="price">$120.00</span>
								</div>
							</div>
							<div class="one-eight text-center">
								<div class="display-tc">
									<a href="#" class="closed"></a>
								</div>
							</div>
						</div>
						<div class="product-cart">
							<div class="one-forth">
								<div class="product-img" style="background-image: url(images/item-7.jpg);">
								</div>
								<div class="display-tc">
									<h3>Product Name</h3>
								</div>
							</div>
							<div class="one-eight text-center">
								<div class="display-tc">
									<span class="price">$68.00</span>
								</div>
							</div>
							<div class="one-eight text-center">
								<div class="display-tc">
									<form action="#">
										<input type="text" name="quantity" class="form-control input-number text-center" value="1" min="1" max="100">
									</form>
								</div>
							</div>
							<div class="one-eight text-center">
								<div class="display-tc">
									<span class="price">$120.00</span>
								</div>
							</div>
							<div class="one-eight text-center">
								<div class="display-tc">
									<a href="#" class="closed"></a>
								</div>
							</div>
						</div>
						<div class="product-cart">
							<div class="one-forth">
								<div class="product-img" style="background-image: url(images/item-8.jpg);">
								</div>
								<div class="display-tc">
									<h3>Product Name</h3>
								</div>
							</div>
							<div class="one-eight text-center">
								<div class="display-tc">
									<span class="price">$68.00</span>
								</div>
							</div>
							<div class="one-eight text-center">
								<div class="display-tc">
									<input type="text" id="quantity" name="quantity" class="form-control input-number text-center" value="1" min="1" max="100">
								</div>
							</div>
							<div class="one-eight text-center">
								<div class="display-tc">
									<span class="price">$120.00</span>
								</div>
							</div>
							<div class="one-eight text-center">
								<div class="display-tc">
		echo							<a href="#" class="closed"></a>
	echo							</div>
	echo						</div>
	echo					"</div>";
	echo				"</div>";
	echo			"</div>";
	echo		"</div>";
	echo"	</div>";


catch(PDOException $e){
			echo $e->getMessage();
		}
		$conn=null;
?>	

	<footer id="colorlib-footer" role="contentinfo">
<div class="container">
<div class="row row-pb-md">
<div class="col-md-3 colorlib-widget">
<h4>About Store</h4>
<p>




</p>
</div>
<div class="col-md-2 colorlib-widget">
<h4>Customer Care</h4>
<p>
<ul class="colorlib-footer-links">
<li><a href="#">Contact</a></li>
<li><a href="#">Returns/Exchange</a></li>
<li><a href="#">Wishlist</a></li>
</ul>
</p>
</div>
<div class="col-md-2 colorlib-widget">
<h4>Information</h4>
<p>
<ul class="colorlib-footer-links">
<li><a href="#">About us</a></li>
<li><a href="#">Terms in Use</a></li>
<li><a href="#">Support</a></li>
</ul>
</p>
</div>

<div class="col-md-2">
<h4>News</h4>
<ul class="colorlib-footer-links">
<li><a href="#">Press</a></li>
<li><a href="#">Exhibitions</a></li>
</ul>
</div>

<div class="col-md-3">
<h4>Contact Information</h4>
<ul class="colorlib-footer-links">
<li> ARTISTERY at JAIPUR ENGINEERING COLLEGE & RESEARCH CENTRE <br> JECRC Campus, Shri Ram ki Nangal<br>
Via Sitapura RIICO<br>
Opp. EPIP Gate, Tonk Road<br>
Jaipur-302022<br>
</li>
<li><a href="tel://+91 8005958377">+91 8005958377</a><br>
<a href="tel://+91 9928820644">+91 9928820644</a></li>
<li><a href="mailto:artistery_star@gmail.com">artistery_star@gmail.com</a></l>
<li><a href="artistery.com">artistery.com</a></li>
</ul>
</div>
</div>
</div>
</footer>
	</div>
	<!-- jQuery -->
	<script src="js/jquery.min.js"></script>
	<!-- jQuery Easing -->
	<script src="js/jquery.easing.1.3.js"></script>
	<!-- Bootstrap -->
	<script src="js/bootstrap.min.js"></script>
	<!-- Waypoints -->
	<script src="js/jquery.waypoints.min.js"></script>
	<!-- Flexslider -->
	<script src="js/jquery.flexslider-min.js"></script>
	<!-- Owl carousel -->
	<script src="js/owl.carousel.min.js"></script>
	<!-- Magnific Popup -->
	<script src="js/jquery.magnific-popup.min.js"></script>
	<script src="js/magnific-popup-options.js"></script>
	<!-- Date Picker -->
	<script src="js/bootstrap-datepicker.js"></script>
	<!-- Stellar Parallax -->
	<script src="js/jquery.stellar.min.js"></script>
	<!-- Main -->
	<script src="js/main.js"></script>
	</script>

	</body>
</html>

