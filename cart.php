<!DOCTYPE html>
<html lang="en">
<head>
<title>Cart</title>
<meta charset="utf-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="description" content="Wish shop project">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" type="text/css" href="styles/bootstrap4/bootstrap.min.css">
<link href="plugins/font-awesome-4.7.0/css/font-awesome.min.css" rel="stylesheet" type="text/css">
<link rel="stylesheet" type="text/css" href="styles/cart.css">
<link rel="stylesheet" type="text/css" href="styles/cart_responsive.css">
</head>
<body>

<div class="super_container">
	
	<!-- Header -->

	<header class="header">
		<div class="header_inner d-flex flex-row align-items-center justify-content-start">
			<div class="logo">
				<p><img src="images/Background/ART.png" height=80 width=250></p>
            </div>
			<nav class="main_nav">
				<ul>
					<li><a href="http://localhost/hack/complete/index.php">home</a></li>
					<li><a href="http://localhost/hack/complete/index.php#handi">Handicrafts</a></li>
					<li><a href="http://localhost/hack/complete/index.php#jewel">Jewellery</a></li>
					<li><a href="http://localhost/hack/complete/index.php#paint">Paintings</a></li>
					<li><a href="contact.php">Contact</a></li>
				</ul>
			</nav>
			<div class="header_content ml-auto">
				<div class="search header_search">
					<form action="#">
						<input type="search" class="search_input" required="required">
						<button type="submit" id="search_button" class="search_button"><img src="images/magnifying-glass.svg" alt=""></button>
					</form>
				</div>
				<div class="shopping">
					<!-- Cart -->
					<a href="#">
						<div class="cart">
							<img src="images/shopping-bag.svg" alt="">
							<div class="cart_num_container">
								<div class="cart_num_inner">
									<div class="cart_num">1</div>
								</div>
							</div>
						</div>
					</a>
					<!-- Star -->
					<a href="#">
						<div class="star">
							<img src="images/star.svg" alt="">
							<div class="star_num_container">
								<div class="star_num_inner">
									<div class="star_num">0</div>
								</div>
							</div>
						</div>
					</a>
					<!-- Avatar -->
					<a href="cart.php">
						<div class="avatar">
							<img src="images/avatar.svg" alt="">
						</div>
					</a>
				</div>
			</div>

			<div class="burger_container d-flex flex-column align-items-center justify-content-around menu_mm"><div></div><div></div><div></div></div>
		</div>
	</header>

	<!-- Menu -->

	<div class="menu d-flex flex-column align-items-end justify-content-start text-right menu_mm trans_400">
		<div class="menu_close_container"><div class="menu_close"><div></div><div></div></div></div>
		<div class="logo menu_mm"><img src="images/Background/ART.png" height=80 width=250></div>
		<div class="search">
			<form action="#">
				<input type="search" class="search_input menu_mm" required="required">
				<button type="submit" id="search_button_menu" class="search_button menu_mm"><img class="menu_mm" src="images/magnifying-glass.svg" alt=""></button>
			</form>
		</div>
		<nav class="menu_nav">
			<ul class="menu_mm">
				<li class="menu_mm"><a href="http://localhost/hack/complete/index.php">Home</a></li>
				<li class="menu_mm"><a href="http://localhost/hack/complete/index.php#handi">Handicraft</a></li>
				<li class="menu_mm"><a href="http://localhost/hack/complete/index.php#jewel">Jewellery</a></li>
				<li class="menu_mm"><a href="http://localhost/hack/complete/index.php#paint">Painting</a></li>
				<li class="menu_mm"><a href="http://localhost/hack/complete/index.phpcontact.php">Contact</a></li>
			</ul>
		</nav>
	</div>

	<!-- Home -->

	<div class="home">
		<div class="home_background parallax-window" data-parallax="scroll" data-image-src="images/wallpaper/w4.jpg" data-speed="0.8"></div>
		<div class="container">
			<div class="row">
				<div class="col">
					<div class="home_container">
						<div class="home_content">
							<div class="home_title">Shopping Cart</div>
							<div class="breadcrumbs">
								<ul>
									<li><a href="index.html">Home</a></li>
									<li>Shopping Cart</li>
								</ul>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>

	<!-- Cart -->
	<div class="cart_container">
		<div class="container">
			<div class="row">
				<div class="col">
					<div class="cart_title">your shopping cart</div>
				</div>
			</div>
<br /><br /><br />
			<?php
		$servername="localhost";
		$user="root";
		$psw="";
		$dbname="cart";
		$conn=mysqli_connect($servername,$user,$psw,$dbname);
		if(!$conn)
		{
			die("connection failed:".mysqli_connect_error());
		}
		try{
			$sql="SELECT * FROM shop";
			$result=$conn->query($sql);
			echo "<center>";
			echo "<table border='1' cellspacing='5' width='95%' ALIGN='CENTER'>";
			echo "<tr>";
				echo "<th>Product Name</th>";
				echo "<th>Price</th>";
				echo "<th>Quantity</th>";
				echo "<th>Total</th>";
			echo "</tr>";
			while($row=$result->fetch_assoc())
			{
				echo "<tr>";
					echo "<td>".$row["Product"]."</td>";
					echo "<td>".$row["Price"]."</td>";
					echo "<td>".$row["Quantity"]."</td>";
					echo "<td>".$row["Total"]."</td>";
				echo "</tr>";
			}
			echo "</table>";
			echo "</center>";
		}
		catch(PDOException $e){
			echo $e->getMessage();
		}
		$conn=null;
	?>

<!--

			<div class="row">
				<div class="col">
					<div class="cart_bar d-flex flex-row align-items-center justify-content-start">
						<div class="cart_bar_title_name">Product</div>
						<div class="cart_bar_title_content ml-auto">
							<div class="cart_bar_title_content_inner d-flex flex-row align-items-center justify-content-end">
								<div class="cart_bar_title_price">Price</div>
								<div class="cart_bar_title_quantity">Quantity</div>
								<div class="cart_bar_title_total">Total</div>
								<div class="cart_bar_title_button"></div>
							</div>
						</div>
					</div>
				</div>
			</div>
 <div class="row">
				<div class="col">
					<div class="cart_products">
						<ul>
							<li class=" cart_product d-flex flex-md-row flex-column align-items-md-center align-items-start justify-content-start">

								
								<div class="cart_product_name"><a href="product.html">2 Piece Swimsuit</a></div>
								<div class="cart_product_info ml-auto">
									<div class="cart_product_info_inner d-flex flex-row align-items-center justify-content-md-end justify-content-start">
									
										<div class="cart_product_price">$35.00</div>
									
										<div class="product_quantity_container">
											<div class="product_quantity clearfix">
												<input id="quantity_input" type="text" pattern="[0-9]*" value="1">
												<div class="quantity_buttons">
													<div id="quantity_inc_button" class="quantity_inc quantity_control"><i class="fa fa-caret-up" aria-hidden="true"></i></div>
													<div id="quantity_dec_button" class="quantity_dec quantity_control"><i class="fa fa-caret-down" aria-hidden="true"></i></div>
												</div>
											</div>
										</div>
										<div class="cart_product_total">$35.00</div>
									
										<div class="cart_product_button">
											<button class="cart_product_remove"><img src="images/trash.png" alt=""></button>
										</div>
									</div>
								</div>
							</li>
						</ul>
					</div>
				</div>
			</div>-->
			<FORM NAME=F1 METHOD=POST ACTION="<?PHP $_SERVER['PHP_SELF']?>">
			<div class="row">
				<div class="col">
					<div class="cart_control_bar d-flex flex-md-row flex-column align-items-start justify-content-start">
						<button class="button_clear cart_button"><input type="submit" onClick="CLICK" name="clear">clear cart</button>
						<button class="button_update cart_button">update cart</button>
						<button class="button_update cart_button_2 ml-md-auto" ><a href="index.php">continue shopping</a></button>
					</div>
				</div>
			</div>
			<div class="row cart_extra">
				<!-- Cart Coupon -->
				<div class="col-lg-6">
					<div class="cart_coupon">
						<div class="cart_title">coupon code</div>
						<form action="#" class="cart_coupon_form d-flex flex-row align-items-start justify-content-start" id="cart_coupon_form">
							<input type="text" class="cart_coupon_input" placeholder="Coupon code">
							<button class="button_clear cart_button_2">apply coupon</button>
						</form>
					</div>
				</div>
				<!-- Cart Total -->
				<div class="col-lg-5 offset-lg-1">
					<div class="cart_total">
						<div class="cart_title">cart total</div>

					<ul>
							<li class="d-flex flex-row align-items-center justify-content-start" >
								
								<div class="cart_total_title">Subtotal</div>
								<div class="cart_total_price ml-auto">rs. 7000</div>
							</li>
							<li class="d-flex flex-row align-items-center justify-content-start">
								<div class="cart_total_title">Shipping</div>
								<div class="cart_total_price ml-auto">rs 50.00</div>
							</li>
							<li class="d-flex flex-row align-items-center justify-content-start">
								<div class="cart_total_title">Total</div>
								<div class="cart_total_price ml-auto">rs. 7050.00</div>
							</li>
						</ul>
						<button class="cart_total_button"><a href="checkout.php">proceed to checkout</a></button>
					</div>
				</div>
			</div>
		</div>
	</div>

	
	<!-- Footer -->

	<footer class="footer">
		<div class="container">
			<div class="row">
				<div class="col text-center">
					<div class="footer_logo"><a href="#">Artistry</a></div>
					<nav class="footer_nav">
						<ul>
							<li><a href="index.html">Home</a></li>
							<li><a href="categories.html">Handicraft</a></li>
							<li><a href="categories.html">Jewellery</a></li>
							<li><a href="categories.html">Paintings</a></li>
							<li><a href="contact.html">contact</a></li>
						</ul>
					</nav>
					<div class="footer_social">
						<ul>
							<li><a href="#"><i class="fa fa-pinterest" aria-hidden="true"></i></a></li>
							<li><a href="#"><i class="fa fa-linkedin" aria-hidden="true"></i></a></li>
							<li><a href="#"><i class="fa fa-instagram" aria-hidden="true"></i></a></li>
							<li><a href="#"><i class="fa fa-reddit-alien" aria-hidden="true"></i></a></li>
							<li><a href="#"><i class="fa fa-twitter" aria-hidden="true"></i></a></li>
						</ul>
					</div>
				</div>
			</div>
		</div>
	</footer>
	<hr color="#DB7115"><hr color="#DB7115">
</div>

<script src="js/jquery-3.2.1.min.js"></script>
<script src="styles/bootstrap4/popper.js"></script>
<script src="styles/bootstrap4/bootstrap.min.js"></script>
<script src="plugins/easing/easing.js"></script>
<script src="plugins/parallax-js-master/parallax.min.js"></script>
<script src="js/cart_custom.js"></script>
<?php
		
IF(ISSET($_POST["clear"]))
{
	CLICK();
}
FUNCTION CLICK()
{
$servername="localhost";
$user="root";
$psw="";
$dbname="cart";
$conn=mysqli_connect($servername,$user,$psw,$dbname);
if(!$conn)
{
die("connection failed:".mysqli_connect_error());
echo "TRy again";
}
$sql="DELETE FROM SHOP;";
$result=mysqli_query($conn,$sql);
$r="select * from shop";
$u=mysqli_query($conn,$r);
$conn->close();
}
?>
</body>
</html>